"""btc_exit ドメインの回帰ガード (決定論・合成 fixture・実取引データ不要)。

主張:
  1. 移植した replay_exit が bot のロジックと一致 (SL→TS→BE 優先・sticky peak・censored)。
  2. 同じ淘汰器が bot 出口候補を裁ける: tighten の本物改善は PASS、緩和候補は高 censored で veto。
  3. load_paths が bot の position_path_*.jsonl を正しく復元する (完了 trade のみ)。
"""
import json
import os
import tempfile

import btc_exit as bx
import domains
import selection_engine as se

GATES = [lambda d, c, i: se.gate_bootstrap(d, c, i, reps=500),
         lambda d, c, i: se.gate_oos(d, c, i, min_n=20),
         lambda d, c, i: se.gate_regime(d, c, i, min_n=20)]


# --- 1. 移植した純関数 replay_exit (bot ロジック一致) ---
def test_replay_sl():
    r = bx.replay_exit([0, -0.5, -0.65], bx.ExitParams(sl_pct=0.6))
    assert (r.reason, r.exit_mtm, r.censored) == ("SL", -0.65, False)


def test_replay_ts():
    r = bx.replay_exit([0, 0.2, 0.5, 0.3], bx.ExitParams(ts_activation_pct=0.15, ts_width_pct=0.175))
    assert r.reason == "TS" and r.exit_mtm == 0.3


def test_replay_be():
    r = bx.replay_exit([0, 0.1, 0.05, 0.01], bx.ExitParams(be_activation_pct=0.08, be_buffer_pct=0.02))
    assert r.reason == "BE" and r.exit_mtm == 0.01


def test_replay_censored_when_no_trigger():
    r = bx.replay_exit([0, 0.05, 0.05], bx.INCUMBENT)
    assert r.reason == "CENSORED" and r.censored is True and r.exit_mtm == 0.05


def test_replay_priority_sl_over_ts():
    # 同一 tick で SL も TS も成立しうる時は SL 優先 (損失保護)
    r = bx.replay_exit([0, 0.3, -0.7], bx.ExitParams(sl_pct=0.6, ts_activation_pct=0.15, ts_width_pct=0.175))
    assert r.reason == "SL" and r.exit_mtm == -0.7


def test_recompute_peak():
    assert bx.recompute_peak([0, 0.2, 0.5, 0.3, 0.1]) == 0.5
    assert bx.recompute_peak([0, -0.1, -0.3]) == 0.0  # 含み益が無ければ peak=0


# --- 2. エンジンが bot 出口候補を裁く (合成) ---
def test_domain_shape_and_slices():
    dom = domains.btc_exit(n=120, seed=1, regime="trend")
    for k in ("variation", "evaluate", "data", "ordered_key", "slice_key"):
        assert k in dom
    assert len({bx.trend_phase(p) for p in dom["data"]} - {"UNKNOWN"}) >= 2


def test_tighten_sl_genuine_improvement_passes():
    # 深い負けを全 regime で浅く切る tighten SL は本物の改善 → 全ゲート PASS
    dom = domains.btc_exit(n=120, seed=1, regime="trend")
    v = se.select(dom, bx.CANDIDATES["tight_sl"], bx.INCUMBENT, gates=GATES)
    assert v.overall == "PASS"


def test_loosen_candidate_flagged_by_censoring():
    # 緩和 (loose TS) は右側打ち切りが多発 = 観測不能 → censored_rate 高 (veto 対象)
    dom = domains.btc_exit(n=120, seed=1, regime="trend")
    cr_loose = bx.censored_rate(dom["data"], bx.CANDIDATES["loose_ts"])
    cr_tight = bx.censored_rate(dom["data"], bx.CANDIDATES["tight_sl"])
    assert cr_loose > 50.0          # 緩和は大半が打ち切り
    assert cr_loose > cr_tight      # tighten より緩和の方が観測不能


def test_censoring_veto_overrides_gate_pass():
    # 核心 (検収#1): 汎用3ゲートが loose を PASS させても、100% censored なら veto が PASS を却下すべき。
    # = ゲート単体では不十分・censoring veto が実仕事をする実証 (bot §I/Lesson-5)。
    paths = bx.make_censoring_trap_paths(n=90, seed=1)
    dom = bx.build_btc_exit_domain(paths, regime_key=bx.trend_phase)
    v = se.select(dom, bx.CANDIDATES["loose_ts"], bx.INCUMBENT, gates=GATES)
    assert v.overall == "PASS"                              # 3ゲートは騙される (本物の改善に見える)
    cr = bx.censored_rate(paths, bx.CANDIDATES["loose_ts"])
    assert cr >= 90.0                                       # だが大半が観測不能
    # 運用ルール: 緩和方向 × censored>30% → veto (gate PASS を上書き)
    vetoed = cr > 30.0
    assert vetoed                                           # → 結論は却下されるべき


def test_synthetic_side_decoupled_from_kind():
    # 検収#2: side が kind と交絡していない (runner=kind2 が両 side に出る)
    paths = bx.make_synthetic_paths(n=120, seed=1)
    runner_sides = {p.side for p in paths if 0.3 < bx.recompute_peak(p.mtm_series) < 0.7
                    and p.mtm_series[-1] < 0.3}
    assert runner_sides == {"LONG", "SHORT"}


def test_equal_candidate_not_passed():
    # incumbent と同一の候補は改善ゼロ → bootstrap が pass しない (ノイズと区別不能)
    dom = domains.btc_exit(n=120, seed=1, regime="trend")
    assert se.gate_bootstrap(dom, bx.INCUMBENT, bx.INCUMBENT, reps=500).status != "pass"


def test_regime_keys_selectable():
    for regime in ("trend", "side", "vol"):
        dom = domains.btc_exit(n=120, seed=1, regime=regime)
        assert callable(dom["slice_key"])


# --- 3. load_paths が実 jsonl を復元 ---
def test_load_paths_parses_completed_only():
    rows = [
        {"event": "open", "trade_id": "T1", "ts_jst": "2026-06-19T03:00:00+09:00",
         "side": "SHORT", "vol_state": "LOW", "mtm_pnl_pct": 0.0,
         "trend_1h": "UP", "trend_4h": "STRONG_UP", "p_up_cal": 0.61},
        {"event": "tick", "trade_id": "T1", "ts_jst": "2026-06-19T03:01:00+09:00", "mtm_pnl_pct": 0.2},
        {"event": "close", "trade_id": "T1", "ts_jst": "2026-06-19T03:02:00+09:00",
         "mtm_pnl_pct": 0.15, "reason": "TS"},
        {"event": "open", "trade_id": "T2", "ts_jst": "2026-06-19T04:00:00+09:00",
         "side": "LONG", "mtm_pnl_pct": 0.0},  # close 無し → 除外
    ]
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "position_path_20260619.jsonl")
        with open(fp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        paths = bx.load_paths(os.path.join(d, "position_path_*.jsonl"))
    assert len(paths) == 1
    p = paths[0]
    assert p.trade_id == "T1" and p.side == "SHORT" and p.mtm_series == [0.0, 0.2, 0.15]
    assert p.actual_final_mtm == 0.15 and p.trend_1h == "UP" and p.p_up_cal == 0.61
    assert p.actual_reason == "TS"          # 実 close 理由を保持 (検収#6・仮想出口との照合用)
    assert bx.trend_phase(p) == "TRENDING_UP"


def test_trend_phase_robust_to_missing_tokens():
    # None/""/"None"/未知トークンは UNKNOWN に倒す (regime ゲートが除外・誤分類しない)
    for bad in ("None", "", "UNKNOWN", "foo"):
        p = bx.PricePath(trade_id="x", trend_1h=bad, trend_4h="UP")
        assert bx.trend_phase(p) == "UNKNOWN"

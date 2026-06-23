"""model_challenge ドメインの回帰ガード (決定論・合成 fixture・実 ML ファイル不要)。

主張:
  1. candidate が真に良い予測 → 全ゲート PASS (true positive)。
  2. candidate の優位が 1 regime だけ → gate_regime が捕まえる (FAIL)。
  3. candidate の優位が 1 時間帯だけ → gate_oos が捕まえる (FAIL)。
  4. ノイズ (優位なし) → gate_bootstrap が捕まえる (FAIL)。
  5. headline_auc が手計算 (Mann-Whitney) と一致。
  6. embargo>0 で gate_oos が縮んだ有効データでも正しく動く (後方互換は別ファイル)。

logloss delta = incumbent_logloss - candidate_logloss を per-sample に gate にかける。
合成は「candidate が label 側に寄った確率を出す = logloss 低い」を regime/時間で制御して埋め込む。
"""
import math

import domains
import model_challenge as mc
import selection_engine as se

GATES = [lambda d, c, i: se.gate_bootstrap(d, c, i, reps=500),
         lambda d, c, i: se.gate_oos(d, c, i, min_n=20),
         lambda d, c, i: se.gate_regime(d, c, i, min_n=20)]

REGIMES = ("UP", "DOWN", "NEUTRAL")
BLOCKS = 3            # ts を3ブロックに分けて oos が効く形に


def _ts(block: int, i: int) -> str:
    """block (0..2) と i から ts 昇順文字列を作る (時系列順を担保)。"""
    return f"2026-06-{10 + block:02d}T{i // 60:02d}:{i % 60:02d}:00"


def _sample(regime: str, block: int, i: int, label: int,
            inc_gap: float, cand_gap: float) -> dict:
    """label 方向に inc_gap / cand_gap だけ寄せた確率を持つ 1 sample を作る。
    gap が大きいほど正解側に確信的=logloss 低い。label=1 なら p=0.5+gap, label=0 なら p=0.5-gap。"""
    def prob(gap):
        g = max(0.0, min(0.49, gap))
        return 0.5 + g if label == 1 else 0.5 - g
    return {"ts": _ts(block, i), "regime": regime, "label": label,
            "incumbent_p": round(prob(inc_gap), 6), "candidate_p": round(prob(cand_gap), 6)}


def _make(advantage):
    """advantage(regime, block) -> candidate の追加 gap (0=同等)。incumbent は常に gap 0.10。
    各 (regime, block) に十分な N を行き渡らせ、label を 50/50 で交互に置く (AUC が偏らない)。"""
    out = []
    per_cell = 14   # 3 regime × 3 block × 14 = 126 sample (各セル/ブロック/regime が min_n=20 を満たす規模)
    idx = 0
    for block in range(BLOCKS):
        for regime in REGIMES:
            for k in range(per_cell):
                label = k % 2                       # 50/50
                adv = advantage(regime, block)
                out.append(_sample(regime, block, idx, label,
                                   inc_gap=0.10, cand_gap=0.10 + adv))
                idx += 1
    return out


# --- (a) candidate が真に良い → PASS ---
def test_genuine_improvement_passes():
    # 全 regime・全 block で candidate がより確信的 (正解側に) → logloss 一様に低い
    samples = _make(lambda r, b: 0.20)
    dom = domains.model_challenge(samples=samples)
    v = se.select(dom, "candidate", "incumbent", gates=GATES)
    assert v.overall == "PASS", {g.name: (g.status, g.numbers) for g in v.gates}


# --- (b) 1 regime だけ優位 → gate_regime FAIL ---
def test_regime_concentrated_advantage_fails_regime_gate():
    # UP だけ candidate が良い・他2 regime は僅かに悪い → 全体平均+ でも regime で符号反転
    def adv(r, b):
        return 0.30 if r == "UP" else -0.02
    samples = _make(adv)
    dom = domains.model_challenge(samples=samples)
    v = se.select(dom, "candidate", "incumbent", gates=GATES)
    g = {x.name: x.status for x in v.gates}
    assert g["regime"] == "fail", {x.name: (x.status, x.numbers) for x in v.gates}
    assert v.overall == "FAIL"


# --- (c) 1 時間帯だけ優位 → gate_oos FAIL ---
def test_time_concentrated_advantage_fails_oos_gate():
    # block 0 だけ candidate が良い・block 1,2 は僅かに悪い → 時系列ブロックで符号反転
    def adv(r, b):
        return 0.30 if b == 0 else -0.02
    samples = _make(adv)
    dom = domains.model_challenge(samples=samples)
    v = se.select(dom, "candidate", "incumbent", gates=GATES)
    g = {x.name: x.status for x in v.gates}
    assert g["oos"] == "fail", {x.name: (x.status, x.numbers) for x in v.gates}
    assert v.overall == "FAIL"


# --- (d) ノイズ (優位なし) → gate_bootstrap FAIL ---
def test_noise_no_advantage_fails_bootstrap():
    # candidate = incumbent と同等 (adv=0) → per-sample delta は全て 0 → CI が 0 を跨ぐ/含む
    samples = _make(lambda r, b: 0.0)
    dom = domains.model_challenge(samples=samples)
    r = se.gate_bootstrap(dom, "candidate", "incumbent", reps=500)
    assert r.status != "pass", (r.status, r.numbers)
    v = se.select(dom, "candidate", "incumbent", gates=GATES)
    assert v.overall == "FAIL"


# --- (e) headline_auc が手計算と一致 ---
def test_headline_auc_matches_manual():
    # 小さく手計算できる集合: candidate は label と完全に順序一致 (AUC=1.0)、
    # incumbent は逆順 (AUC=0.0)。tie 無しの素直なケース。
    samples = [
        {"ts": "t1", "regime": "UP", "label": 1, "incumbent_p": 0.2, "candidate_p": 0.9},
        {"ts": "t2", "regime": "UP", "label": 0, "incumbent_p": 0.8, "candidate_p": 0.1},
        {"ts": "t3", "regime": "UP", "label": 1, "incumbent_p": 0.3, "candidate_p": 0.8},
        {"ts": "t4", "regime": "UP", "label": 0, "incumbent_p": 0.7, "candidate_p": 0.2},
    ]
    h = mc.headline_auc(samples)
    assert h["candidate_auc"] == 1.0    # 正例が全部 高スコア
    assert h["incumbent_auc"] == 0.0    # 正例が全部 低スコア
    assert h["delta_auc"] == 1.0
    assert h["n"] == 4

    # tie を含むケースを Mann-Whitney 手計算と照合
    # scores=[0.5,0.5,0.9], labels=[0,1,1]: 0.5同値は平均順位1.5、0.9は順位3
    #   sum_pos_ranks = 1.5(label1の0.5) + 3(0.9) = 4.5 ; pos=2,neg=1
    #   AUC = (4.5 - 2*3/2) / (2*1) = (4.5-3)/2 = 0.75
    tied = [
        {"ts": "u1", "regime": "UP", "label": 0, "incumbent_p": 0.5, "candidate_p": 0.5},
        {"ts": "u2", "regime": "UP", "label": 1, "incumbent_p": 0.5, "candidate_p": 0.5},
        {"ts": "u3", "regime": "UP", "label": 1, "incumbent_p": 0.5, "candidate_p": 0.9},
    ]
    h2 = mc.headline_auc(tied)
    assert h2["candidate_auc"] == 0.75


# --- (e2) logloss / sample_delta の符号と安全 clip ---
def test_logloss_and_delta_sign():
    # candidate が正解側に確信的 → logloss 低 → delta>0
    s = {"label": 1, "incumbent_p": 0.6, "candidate_p": 0.9}
    assert mc.logloss(0.9, 1) < mc.logloss(0.6, 1)
    assert mc.sample_delta(s) > 0
    # clip: p=0 や p=1 でも有限 (log(0) を防ぐ)
    assert math.isfinite(mc.logloss(0.0, 1))
    assert math.isfinite(mc.logloss(1.0, 0))


# --- (f) embargo>0 で gate_oos が縮んだ有効データでも正しく動く ---
def test_oos_with_embargo_still_works():
    # 全 block で改善あり (genuine) → embargo で各ブロック端を捨てても sign は + のまま PASS
    samples = _make(lambda r, b: 0.20)
    dom = domains.model_challenge(samples=samples)
    base = se.gate_oos(dom, "candidate", "incumbent", min_n=10, embargo=0)
    emb = se.gate_oos(dom, "candidate", "incumbent", min_n=10, embargo=5)
    assert base.status == "pass" and emb.status == "pass"
    # embargo で各ブロックの実効 N が縮む (=境界 sample を捨てている証拠)
    # min_n を上げると embargo 版が先に insufficient になりうる
    big = se.gate_oos(dom, "candidate", "incumbent", min_n=35, embargo=8)
    assert big.status in ("pass", "insufficient")  # 縮んでも符号は壊れない


# --- ドメイン形・load_predictions ---
def test_domain_shape():
    samples = _make(lambda r, b: 0.20)
    dom = domains.model_challenge(samples=samples)
    for k in ("variation", "evaluate", "data", "ordered_key", "slice_key"):
        assert k in dom
    # evaluate は負の logloss (大=良)・candidate>incumbent
    s = dom["data"][0]
    assert dom["evaluate"]("candidate", s) >= dom["evaluate"]("incumbent", s) - 1e-9
    # ts 昇順
    keys = [dom["ordered_key"](s) for s in dom["data"]]
    assert keys == sorted(keys)


def test_load_predictions_parses_and_sorts(tmp_path):
    import json
    rows = [
        {"ts": "2026-06-12T00:00:00", "regime": "DOWN", "label": 0,
         "incumbent_p": 0.4, "candidate_p": 0.2},
        {"ts": "2026-06-10T00:00:00", "regime": "UP", "label": 1,
         "incumbent_p": 0.6, "candidate_p": 0.9},          # 後で先頭に来るべき (ts 昇順)
        {"ts": "2026-06-11T00:00:00", "label": 1},          # 必須キー欠 → skip
        "{ broken json",                                     # 壊れ JSON → skip
    ]
    fp = tmp_path / "eval_predictions_20260612.jsonl"
    with open(fp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write((json.dumps(r) if isinstance(r, dict) else r) + "\n")
    samples = mc.load_predictions(str(tmp_path / "eval_predictions_*.jsonl"))
    assert len(samples) == 2
    assert samples[0]["ts"] == "2026-06-10T00:00:00"        # 昇順
    assert samples[0]["regime"] == "UP" and samples[0]["label"] == 1

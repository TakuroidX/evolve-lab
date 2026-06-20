"""btc_exit.py — selection_engine 用「BTC bot 出口リプレイ」ドメイン (self-contained・移植版)。

狙い (DESIGN.md step5): 淘汰エンジンが**生まれ故郷の bot 自身の出口**を裁けることを示す
= 「bot はこのエンジンの一例」。bot の純関数 replay_exit/load_paths を **最小移植** し
(出典: Bitflyer-Trading-Bot-Pro_V2 tools/exit_replay/engine.py, 2026-06-14)、evolve-lab を
独立・依存ゼロのまま保つ (bot リポジトリに結合しない)。

データの扱い (公開札の方針, interview 2026-06-21):
  - 公開リポ/CI = 決定論の **合成 path fixture** (make_synthetic_paths)。実取引履歴は焼かない。
  - ローカル = load_paths(path_glob) で bot の実 logs/position_path_*.jsonl を **read-only** 評価。
    README には実走の数字だけ正直に引用する (raw path は公開しない)。

ドメイン写像:
  - evaluate(exit_params, path) = replay_exit(path.mtm_series, exit_params).exit_mtm (% 値・大=良)
  - select(dom, candidate_params, incumbent_params) で 1候補ずつ判定 (A/B と同じく select 専用)。
  - ordered_key = trade_id (時系列) / slice_key = regime (trend_phase / side / vol)

🔴 censoring (bot §I): path は実 close で右側打ち切り。**緩和 (利を伸ばす) 方向は原理的に観測不能**で
   tighten しか測れない。汎用3ゲートは censoring を見ないので、censored_rate を別途surfaceし
   緩和候補の結論は veto する (DESIGN step6 で汎用ゲート化予定)。make_censoring_trap_paths() は
   「汎用ゲートが PASS でも 100% censored で veto すべき」場面を実演する (veto が実仕事をする証拠)。

正直な scope (検収レビュー 2026-06-21 反映):
  - **忠実移植 (logic 不変)**: replay_exit / recompute_peak / load_paths (bot engine.py からバイト一致)。
  - **evolve-lab 側の新規** (移植でない): trend_phase / by_side / by_vol の regime キー。bot の trend_phase は
    fitness.py 側にあり別物。だから regime 分類の細部は bot と一致を主張しない。
  - 淘汰は**汎用3ゲート (bootstrap平均CI / OOS / regime) + censoring surface** = bot fitness.py の
    median/外れ値/payoff/censoring ゲートの「再表現」であって gate 同一ではない。実 175-path 実走で
    判定が**一致**する、が主張 (gate 同一ではない)。
純Python・依存ゼロ・決定論 (seed 固定)。
"""
from __future__ import annotations

import glob
import json
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# データモデル + 純関数 (bot tools/exit_replay/engine.py から最小移植・ロジック不変)
# ---------------------------------------------------------------------------
@dataclass
class PricePath:
    """1 trade の保有中 mtm 経路 (open→close, % 値)。"""
    trade_id: str
    side: str = ""                  # LONG / SHORT (regime スライス用)
    vol_state: str = "NORMAL"       # LOW / NORMAL / HIGH (regime スライス用)
    mtm_series: list = field(default_factory=list)  # open(=0.0)→…→close の mtm%
    actual_final_mtm: float = 0.0
    actual_reason: str = ""         # 実 close 理由 (SL/TS/BE 等・仮想出口との照合用)
    trend_1h: str = "UNKNOWN"
    trend_4h: str = "UNKNOWN"
    p_up_cal: float = 0.5


@dataclass(frozen=True)
class ExitParams:
    """仮想出口パラメータ (全て % 値、0 で当該ルール無効)。bot ExitParams と同形。"""
    sl_pct: float = 0.0
    ts_activation_pct: float = 0.0
    ts_width_pct: float = 0.0
    be_activation_pct: float = 0.0
    be_buffer_pct: float = 0.0


@dataclass
class ExitResult:
    exit_idx: int
    exit_mtm: float
    reason: str        # SL / TS / BE / CENSORED
    censored: bool     # True = 実 close まで未トリガ (緩和方向の根拠に使えない)


def recompute_peak(mtm_series: list) -> float:
    """記録 peak を信用せず mtm 系列から sticky peak を再計算。"""
    peak = 0.0
    for m in mtm_series:
        if m > peak:
            peak = m
    return peak


def replay_exit(mtm_series: list, params: ExitParams) -> ExitResult:
    """mtm 系列に仮想出口を適用し最早トリガを返す (純関数)。優先: SL→TS→BE。
    どれも実 close まで未トリガ → CENSORED。bot engine.py replay_exit とロジック同一。"""
    if not mtm_series:
        return ExitResult(0, 0.0, "CENSORED", True)
    peak = 0.0
    ts_armed = False
    be_armed = False
    for i, m in enumerate(mtm_series):
        if m > peak:
            peak = m
        if params.sl_pct and m <= -params.sl_pct:
            return ExitResult(i, m, "SL", False)
        if params.ts_activation_pct and peak >= params.ts_activation_pct:
            ts_armed = True
        if ts_armed and params.ts_width_pct and m <= peak - params.ts_width_pct:
            return ExitResult(i, m, "TS", False)
        if params.be_activation_pct and peak >= params.be_activation_pct:
            be_armed = True
        if be_armed and m <= params.be_buffer_pct:
            return ExitResult(i, m, "BE", False)
    return ExitResult(len(mtm_series) - 1, mtm_series[-1], "CENSORED", True)


def load_paths(path_glob: str) -> list:
    """bot の position_path_*.jsonl (glob) を完了 trade ごとの PricePath に復元 (read-only・最小移植)。
    trend/p_up_cal は open event の実値 (#15 v2)。close レコードのある trade のみ。"""
    by: dict = {}
    for fp in sorted(glob.glob(path_glob)):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = r.get("trade_id")
                if tid:
                    by.setdefault(tid, []).append(r)
    out = []
    for tid, recs in by.items():
        if not any(r.get("event") == "close" for r in recs):
            continue
        recs.sort(key=lambda r: r.get("ts_jst", ""))
        series = [float(r.get("mtm_pnl_pct", 0.0)) for r in recs]
        close = [r for r in recs if r.get("event") == "close"][-1]
        op = next((r for r in recs if r.get("event") == "open"), recs[0])
        out.append(PricePath(
            trade_id=tid, side=recs[0].get("side", ""),
            vol_state=recs[0].get("vol_state", "NORMAL"), mtm_series=series,
            actual_final_mtm=float(close.get("mtm_pnl_pct", 0.0)),
            actual_reason=str(close.get("reason", "")),
            trend_1h=str(op.get("trend_1h") or "UNKNOWN"),
            trend_4h=str(op.get("trend_4h") or "UNKNOWN"),
            p_up_cal=float(op.get("p_up_cal", 0.5))))
    out.sort(key=lambda p: p.trade_id)
    return out


# ---------------------------------------------------------------------------
# regime スライスキー (regime ゲート用)
# ---------------------------------------------------------------------------
_KNOWN_TREND = {"UP", "STRONG_UP", "DOWN", "STRONG_DOWN", "NEUTRAL"}


def trend_phase(p: PricePath) -> str:
    """trend_1h/4h を相場局面に (evolve-lab 側の regime キー・bot fitness.py の trend_phase とは別物)。
    両方 UP系→TRENDING_UP / 両方 DOWN系→TRENDING_DOWN / 未知トークン(None/""/"None"含む)→UNKNOWN
    (regime ゲートが除外) / それ以外→MIXED。known トークンのみ信用 (誤分類を防ぐ)。"""
    t1, t4 = p.trend_1h.upper(), p.trend_4h.upper()
    if t1 not in _KNOWN_TREND or t4 not in _KNOWN_TREND:
        return "UNKNOWN"
    up = sum(1 for t in (t1, t4) if "UP" in t)
    dn = sum(1 for t in (t1, t4) if "DOWN" in t)
    if up == 2:
        return "TRENDING_UP"
    if dn == 2:
        return "TRENDING_DOWN"
    return "MIXED"


def by_side(p: PricePath) -> str:
    return p.side or "UNKNOWN"


def by_vol(p: PricePath) -> str:
    return p.vol_state or "UNKNOWN"


# ---------------------------------------------------------------------------
# bot 現行出口 + 代表候補 (bot run_fitness_panel.py と同値・1:1 クロスチェック用)
# ---------------------------------------------------------------------------
INCUMBENT = ExitParams(sl_pct=0.6, ts_activation_pct=0.15, ts_width_pct=0.175,
                       be_activation_pct=0.08, be_buffer_pct=0.02)
CANDIDATES = {
    "tight_ts": ExitParams(sl_pct=0.6, ts_activation_pct=0.10, ts_width_pct=0.05,
                           be_activation_pct=0.08, be_buffer_pct=0.02),   # 早利確
    "tight_sl": ExitParams(sl_pct=0.4, ts_activation_pct=0.15, ts_width_pct=0.175,
                           be_activation_pct=0.08, be_buffer_pct=0.02),   # 早損切り
    "loose_ts": ExitParams(sl_pct=0.6, ts_activation_pct=0.15, ts_width_pct=0.35,
                           be_activation_pct=0.08, be_buffer_pct=0.02),   # 利伸ばし(緩和)
}


# ---------------------------------------------------------------------------
# 合成 path fixture (公開/CI 用・実取引データを焼かない・決定論)
# ---------------------------------------------------------------------------
def _down_series(depth: float) -> list:
    """単調下落して -depth で close (深い負け = SL の効く題材)。"""
    steps = max(2, int(depth / 0.1) + 1)
    return [round(-depth * k / steps, 4) for k in range(steps + 1)]


def _runner_giveback(peak: float, close: float) -> list:
    """+peak まで上げてから close まで吐き出す (TS の効く題材)。"""
    up = [round(peak * k / 5, 4) for k in range(6)]            # 0→peak
    down = [round(peak - (peak - close) * k / 4, 4) for k in range(1, 5)]  # peak→close
    return up + down


def make_synthetic_paths(n: int = 120, seed: int = 1) -> list:
    """決定論の合成 path。深い負け / runner吐き出し / 小勝ち を混在させ、side×trend_phase
    の各スライスに負けを行き渡らせる (tighten が全 regime で効く=PASS を作れる形)。"""
    out = []
    phases = [("STRONG_UP", "UP"), ("STRONG_DOWN", "DOWN"), ("UP", "DOWN")]  # UP/DOWN/MIXED
    for i in range(n):
        rng = random.Random(seed * 10_000 + i)
        side = "LONG" if rng.random() < 0.5 else "SHORT"  # kind と独立に引く (側×種の交絡を避ける・検収#2)
        t1, t4 = phases[i % 3]
        kind = i % 4
        if kind in (0, 1):                       # 深い負け (-0.45..-0.65): tighten SL が効く
            series = _down_series(0.45 + rng.random() * 0.20)
        elif kind == 2:                          # runner 吐き出し: tighten TS が効く
            series = _runner_giveback(0.40 + rng.random() * 0.20, rng.uniform(-0.05, 0.10))
        else:                                    # 小勝ち: 出口で大差つかない
            series = [0.0, 0.05, 0.10, 0.12, 0.10, 0.08]
        out.append(PricePath(
            trade_id=f"SYN{i:04d}", side=side, vol_state="LOW", mtm_series=series,
            actual_final_mtm=series[-1], trend_1h=t1, trend_4h=t4, p_up_cal=0.5))
    return out


def make_censoring_trap_paths(n: int = 90, seed: int = 1) -> list:
    """censoring veto が**実仕事をする**ことを実演する fixture (検収#1)。
    各 path = 押し目→高値 close。incumbent TS は押し目で早期発火するが、loose TS は close まで乗り
    高値を取る → 汎用3ゲートは loose を「本物の改善」として PASS させる。だが loose は 100% censored
    (実 close から先は観測不能) → **gate が PASS でも censoring veto が PASS を却下すべき場面** (bot §I/Lesson-5)。"""
    out = []
    phases = [("STRONG_UP", "UP"), ("STRONG_DOWN", "DOWN"), ("UP", "DOWN")]
    for i in range(n):
        rng = random.Random(seed * 9_000 + i)
        p1 = 0.35 + rng.random() * 0.15            # 初期 peak
        dip = p1 - 0.20 - rng.random() * 0.05      # incumbent(width 0.175)が発火する押し
        close = p1 + 0.15 + rng.random() * 0.15    # その後の高値 close
        series = [0.0, round(p1 / 2, 4), round(p1, 4), round(dip, 4),
                  round((dip + close) / 2, 4), round(close, 4)]
        side = "LONG" if rng.random() < 0.5 else "SHORT"
        t1, t4 = phases[i % 3]
        out.append(PricePath(
            trade_id=f"TRAP{i:04d}", side=side, vol_state="LOW", mtm_series=series,
            actual_final_mtm=series[-1], trend_1h=t1, trend_4h=t4, p_up_cal=0.5))
    return out


# ---------------------------------------------------------------------------
# ドメイン構築 + censoring surface
# ---------------------------------------------------------------------------
def build_btc_exit_domain(paths: list, regime_key=trend_phase) -> dict:
    """select 専用 domain。candidate/incumbent は ExitParams を select() に直接渡す。"""
    return {
        "variation": lambda params, rng: params,          # 未使用 (出口は進化でなく候補比較)
        "evaluate": lambda params, p: replay_exit(p.mtm_series, params).exit_mtm,
        "data": paths,
        "ordered_key": lambda p: p.trade_id,
        "slice_key": regime_key,
    }


def censored_rate(paths: list, params: ExitParams) -> float:
    """候補 params が実 close まで未トリガ (右側打ち切り) になる割合。
    高い = 緩和方向で観測不能 = 汎用ゲートの結論を信用してはいけない (censoring veto)。"""
    if not paths:
        return 0.0
    c = sum(1 for p in paths if replay_exit(p.mtm_series, params).censored)
    return round(100 * c / len(paths), 1)

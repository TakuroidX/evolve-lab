"""selection_engine.py — ドメイン非依存の「信頼できる淘汰エンジン」。

BTC bot の exit fitness harness (tools/exit_replay/fitness.py) と、evolve-lab の進化ループ
から **核だけを抽出**した汎用オブジェクト。変異 × 淘汰(自分を騙さないゲート) × 遺伝 を、
どのドメインにも plug できる1つの object にする。

核の主張 (evolve-lab / bot 1.5年で実証):
  進化の質は fitness の質で決まる。平坦/ノイズ地形で変異を回すと過学習ドリフトにしかならない。
  だから「**変異の前に、信頼できる淘汰を**」。淘汰 = 自分を騙さないゲート群。
  bot で実証 (2026-06-19): 同じゲートが自分の PASS すら淘汰し、署名付きで「エッジ無し」を出せた。

Domain は dict で与える (重い抽象化を避ける):
  variation(parent, rng) -> candidate         # 変異
  evaluate(candidate, sample) -> float        # 1 sample の良さ (大きいほど良い)
  data: list[sample]                          # held-out 評価標本 (選別が触れてよい・test は別管理)
  ordered_key(sample) -> sortable  (任意)      # OOS 時系列分割の順序キー (既定: 元の並び)
  slice_key(sample) -> hashable    (任意)      # regime 横断スライスのキー ("UNKNOWN" は除外)

純Python・依存ゼロ・決定論 (seed 固定)。LLM 不使用。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class GateResult:
    name: str
    status: str  # "pass" | "fail" | "insufficient"
    reason: str
    numbers: dict = field(default_factory=dict)


@dataclass
class Verdict:
    overall: str  # "PASS" | "HOLD" | "FAIL"
    gates: list


# --- per-sample デルタ = candidate が incumbent よりどれだけ良いか (大きいほど改善) ---
def _deltas(domain, candidate, incumbent):
    ev = domain["evaluate"]
    return [ev(candidate, s) - ev(incumbent, s) for s in domain["data"]]


# --- 汎用ゲート (どのドメインでも使える3本) ---
def gate_bootstrap(domain, candidate, incumbent, reps: int = 2000, seed: int = 0):
    """per-sample 改善デルタ平均の 95%CI が 0 超か (ノイズフロアを超える本物の改善か)。"""
    d = _deltas(domain, candidate, incumbent)
    n = len(d)
    if n < 10:
        return GateResult("bootstrap", "insufficient", f"N={n}<10 でブートストラップ不能", {"n": n})
    rng = random.Random(seed)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    lo, hi = means[int(reps * 0.025)], means[int(reps * 0.975)]
    nums = {"n": n, "mean": round(sum(d) / n, 5), "ci95": [round(lo, 5), round(hi, 5)]}
    if lo > 0:
        return GateResult("bootstrap", "pass", "95%CI 下限>0 (ノイズ超えの改善)", nums)
    if hi < 0:
        return GateResult("bootstrap", "fail", "95%CI 上限<0 (有意に悪化)", nums)
    return GateResult("bootstrap", "fail", "95%CI が 0 を跨ぐ (ノイズと区別不能)", nums)


def gate_oos(domain, candidate, incumbent, n_folds: int = 3, min_n: int = 20):
    """時系列 k-fold: 順序で n_folds ブロックに分け、どの期間でも改善が正のままか (期間過学習を弾く)。"""
    key = domain.get("ordered_key") or (lambda s: 0)
    data = sorted(domain["data"], key=key)
    n = len(data)
    blocks = [data[i * n // n_folds:(i + 1) * n // n_folds] for i in range(n_folds)]
    qual = [b for b in blocks if len(b) >= min_n]
    if len(qual) < 2:
        return GateResult("oos", "insufficient",
                          f"N>={min_n} の時間ブロックが {len(qual)}個 (<2)", {"sizes": [len(b) for b in blocks]})
    ev = domain["evaluate"]
    md = [sum(ev(candidate, s) - ev(incumbent, s) for s in b) / len(b) for b in qual]
    signs = {1 if x > 0 else (-1 if x < 0 else 0) for x in md}
    nums = {"per_block": [round(x, 5) for x in md]}
    if signs == {1}:
        return GateResult("oos", "pass", "全ブロックで改善+ (期間過学習でない)", nums)
    return GateResult("oos", "fail", "ブロック間で符号反転 (一部期間だけの改善=期間過学習)", nums)


def gate_regime(domain, candidate, incumbent, min_n: int = 20):
    """regime スライス横断で改善の符号が反転しないか。"UNKNOWN" スライスは本物として数えない。"""
    key = domain.get("slice_key")
    if key is None:
        return GateResult("regime", "insufficient", "slice_key 未定義 (このドメインに regime 軸なし)", {})
    groups: dict = {}
    for s in domain["data"]:
        groups.setdefault(key(s), []).append(s)
    qual = {k: v for k, v in groups.items()
            if len(v) >= min_n and k is not None and str(k).upper() != "UNKNOWN"}
    if len(qual) < 2:
        return GateResult("regime", "insufficient",
                          f"N>={min_n} の (UNKNOWN除く) slice が {len(qual)}個 (<2)",
                          {"sizes": {str(k): len(v) for k, v in groups.items()}})
    ev = domain["evaluate"]
    md = {k: sum(ev(candidate, s) - ev(incumbent, s) for s in v) / len(v) for k, v in qual.items()}
    signs = {1 if x > 0 else (-1 if x < 0 else 0) for x in md.values()}
    nums = {"per_slice": {str(k): round(x, 5) for k, x in md.items()}}
    if signs == {1}:
        return GateResult("regime", "pass", "全 slice で改善+ (regime 横断で頑健)", nums)
    return GateResult("regime", "fail", "slice 間で符号反転 (この相場だけの改善=過学習疑い)", nums)


DEFAULT_GATES = [gate_bootstrap, gate_oos, gate_regime]


# --- 淘汰 (scorecard ゲート。ranking しない = bot Lesson-5) ---
def select(domain, candidate, incumbent, gates=None) -> Verdict:
    """候補を incumbent と比較し、全ゲートのスコアカードで PASS/HOLD/FAIL を返す。

    1つでも fail → FAIL。fail 無し & insufficient あり → HOLD。全 pass → PASS。
    PASS でも「実装してよい」ではない (bot Lesson-6: PASS≠trustworthy。最終判断は人/critic)。
    """
    gates = gates or DEFAULT_GATES
    results = [g(domain, candidate, incumbent) for g in gates]
    st = [r.status for r in results]
    overall = "FAIL" if "fail" in st else ("HOLD" if "insufficient" in st else "PASS")
    return Verdict(overall, results)


# --- 進化ループ (変異 × 淘汰 × 遺伝) ---
def evolve(domain, incumbent, generations: int = 500, seed: int = 1,
           accept_gates=None):
    """各世代: 変異 → (軽い) 淘汰ゲートで受理判定 → 受理なら遺伝 (次世代の親)。

    ループ内の受理は軽いゲート (既定 = bootstrap のみ) で回し、最終的な ship 判断は別途
    select() のフルスコアカード + 人/critic で行う (bot の運用と同型: ループは安く、出荷は厳格)。
    """
    accept_gates = accept_gates or [gate_bootstrap]
    rng = random.Random(seed)
    var = domain["variation"]
    inc = incumbent
    accepted = 0
    for _ in range(generations):
        cand = var(inc, rng)
        if select(domain, cand, inc, accept_gates).overall == "PASS":
            inc = cand
            accepted += 1
    return {"incumbent": inc, "accepted": accepted}

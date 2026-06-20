"""ab_select.py — selection_engine 用「A/B 選別の規律」ドメイン (決定論・API 不要)。

狙い (DESIGN.md Domain 3): A/B テストで「平均が高い方を採用」(ナイーブ選別) が出荷してしまう偽の勝ちを、
同じ淘汰器の3ゲートが**別々の失敗モードとして**捕まえることを、正解を埋め込んだ合成データで実証する。
prompt_opt と違い A/B は「処理 vs 対照」の**比較**なので evolve はせず select() を使う (select 専用ドメイン)。

  - sample = 1ユニット観測 {unit, week, segment, a, b}。a=対照(incumbent)指標 / b=処理(candidate)指標。
    合成なので両方の counterfactual を持つ (=対応のある delta b-a)。実 A/B は非対応で two-sample 版に
    なるが、ゲートの**論理は同一** (ノイズフロア / 期間横断 / セグメント横断で符号が割れないか)。
  - evaluate(variant, sample) = sample[variant] (variant ∈ {"a","b"})。select(dom,"b","a") で B vs A。
  - ordered_key=week (OOS: peeking/季節性) / slice_key=segment (regime: 1セグメント偏重)。

4 シナリオ (正解を埋め込み):
  genuine      : 全週・全セグメントで真の改善 (+) → 全ゲート PASS (=真陽性。ナイーブも正しい=正直に併記)
  peeking      : 初週は大きく+、後週は- (純効果は僅か+) → oos 却下 (期間ブロックで符号反転) ※構造的・seed頑健
  concentrated : 1セグメントだけ大勝ち・他は- (全体平均は+) → regime 却下 (全体ロールアウト不当) ※構造的・seed頑健
  noise        : 真の効果≈0 (僅かな drift+高分散)。観測平均が+に転べば naive は出荷するが (seed依存・~8割)、
                 bootstrap は 0 と区別できず却下 (~95%)。**確率的**=「4/4 必ず」とは言わない (ノイズの本質)。
                 構造的偽陽性 peeking/concentrated とは性質が違う点を誇張しない。
純Python・依存ゼロ・決定論 (seed 固定)。
"""
from __future__ import annotations

import random

SEGMENTS = ["new", "returning", "mobile"]
WEEKS = 6
SCENARIOS = ("genuine", "noise", "peeking", "concentrated")


def _effect(scenario: str, week: int, seg: str, rng: random.Random) -> float:
    """処理効果 b-a の真値 (+ノイズ)。シナリオごとに「どこで割れるか」を埋め込む。"""
    if scenario == "genuine":
        return 5.0 + rng.gauss(0, 3)                       # どこでも + (本物)
    if scenario == "noise":
        return 0.3 + rng.gauss(0, 8)                       # 真値≈0・高分散 → CI が 0 を跨ぐ
    if scenario == "peeking":
        return (3 - week) * 4.0 + rng.gauss(0, 3)          # 初週+12 … 後週-8 (週で符号反転)
    if scenario == "concentrated":
        base = {"new": 10.0, "returning": -2.0, "mobile": -2.0}[seg]  # new だけ大勝ち
        return base + rng.gauss(0, 3)
    raise ValueError(f"unknown scenario: {scenario}")


def make_ab_samples(scenario: str, n: int = 120, seed: int = 1) -> list:
    """week と segment を独立・均等配置 (OOS ブロック / regime スライスが偏らないように)。
    n>=60 必須: oos 3ブロック × regime 3スライスが各 min_n=20 を満たすため
    (未満だとゲートが insufficient になり FAIL が HOLD に化け、偽陽性が見逃される)。"""
    if n < 60:
        raise ValueError(f"n>=60 必須 (oos3ブロック×regime3スライス×min_n=20)。received n={n}")
    rng = random.Random(seed)
    out = []
    for i in range(n):
        seg = SEGMENTS[i % len(SEGMENTS)]
        week = (i // len(SEGMENTS)) % WEEKS
        a = rng.gauss(100, 10)                  # 対照指標
        b = a + _effect(scenario, week, seg, rng)  # 処理指標 (対応のある counterfactual)
        out.append({"unit": i, "week": week, "segment": seg, "a": a, "b": b})
    return out


def build_ab_domain(samples: list) -> dict:
    """select() 専用 domain。variation は no-op (A/B は比較であって進化でない)。"""
    return {
        "variation": lambda variant, rng: variant,   # 未使用 (evolve しない)
        "evaluate": lambda variant, s: s[variant],   # variant ∈ {"a","b"}
        "data": samples,
        "ordered_key": lambda s: s["week"],          # OOS: 週
        "slice_key": lambda s: s["segment"],         # regime: セグメント
    }


def naive_winner(domain: dict) -> dict:
    """ナイーブ選別 = 平均が高い方を採用 (現場が普通にやること)。B の平均>A なら出荷。"""
    data = domain["data"]
    mean_a = sum(s["a"] for s in data) / len(data)
    mean_b = sum(s["b"] for s in data) / len(data)
    return {"ships_b": mean_b > mean_a, "mean_diff": round(mean_b - mean_a, 3)}

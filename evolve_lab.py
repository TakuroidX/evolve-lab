"""evolve-lab — 「変異 × 信頼できる淘汰 × 遺伝」を、勾配が本物のドメインで動かす最小POC。

背景: BTC bot は薄エッジ(=勾配ほぼ無)のため、進化ループは過学習ドリフトになり proposer を
意図停止した。本POCは「勾配が在るドメインなら、同じループ+淘汰器が実際に登る。そして
淘汰器ありは汎化し、ナイーブ淘汰(trainだけ見る=1y3mループ型)は過学習する」を対比実証する。
= bot が進化しなかったのは"方法"でなく"市場に勾配が無い"せい、を示す。

ドメイン: 記号回帰。隠れた真の関数 g(x) (=本物の勾配/信号) を、ノイズ付きデータから
多項式係数を進化させて近似する。複雑な係数は train のノイズに過学習しうる(=市場の罠の縮図)。

淘汰器 (BTC bot の fitness.py の思想を一般化):
  - bootstrap: train の誤差削減の per-point デルタ平均 95%CI が 0 超 (ノイズフロア超え)
  - OOS holdout: 改善が validation(未使用)でも正で残る (期間/標本過学習を弾く)
test(完全未使用)は最終報告にのみ使い、選別には一切触らせない。

純Python・依存ゼロ・決定論(seed固定)。bot リポジトリと無関係の別プロジェクト。
"""
from __future__ import annotations

import random


# --- 真の信号 (この世界には勾配が在る ── 市場と違って) ---
def true_g(x: float) -> float:
    return 0.5 * x * x - 2.0 * x + 1.0  # 隠れた次数2の関数


def make_dataset(n: int, noise: float, seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [(x, true_g(x) + rng.gauss(0, noise))
            for x in (rng.uniform(-3, 3) for _ in range(n))]


# --- 候補モデル: 多項式係数 (進化対象)。次数4 = 次数2の真関数に対し過学習の余地あり ---
def predict(coefs: list[float], x: float) -> float:
    return sum(c * x ** i for i, c in enumerate(coefs))


def mse(coefs: list[float], data: list[tuple[float, float]]) -> float:
    return sum((predict(coefs, x) - y) ** 2 for x, y in data) / len(data)


# --- 変異 ---
def mutate(coefs: list[float], rng: random.Random, scale: float = 0.25) -> list[float]:
    c = list(coefs)
    i = rng.randrange(len(c))
    c[i] += rng.gauss(0, scale)
    return c


# --- 信頼できる淘汰 (一般化版) ---
def _per_point_delta(cand, inc, data) -> list[float]:
    # 正 = candidate の方が誤差が小さい (改善)
    return [(predict(inc, x) - y) ** 2 - (predict(cand, x) - y) ** 2 for x, y in data]


def _bootstrap_lo(deltas: list[float], reps: int, seed: int) -> float:
    n = len(deltas)
    rng = random.Random(seed)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return means[int(reps * 0.025)]  # 95%CI 下限


def harness_accepts(cand, inc, train, val, reps: int = 1500, seed: int = 0):
    """淘汰器: 改善が held-out(val) 上で統計的に本物か = bootstrap CI 下限>0。
    train でいくら良くても val で有意改善しなければ却下 = 過学習(ノイズ適合)を弾く (OOS+bootstrap 融合)。"""
    return _bootstrap_lo(_per_point_delta(cand, inc, val), reps, seed) > 0


def naive_accepts(cand, inc, train, val=None):
    """ナイーブ淘汰 (=1y3mループ型): train が改善すれば即採用。OOS を見ない。"""
    return mse(cand, train) < mse(inc, train)


# --- 進化ループ (変異 × 淘汰 × 遺伝) ---
def evolve(accept, generations: int = 1500, seed: int = 1):
    rng = random.Random(seed)
    train = make_dataset(12, noise=1.0, seed=seed)         # 少標本 = 過学習の余地大
    val = make_dataset(60, noise=1.0, seed=seed + 7)        # 淘汰器の OOS 用 (選別に使う)
    test = make_dataset(400, noise=1.0, seed=seed + 9999)   # 完全未使用・最終報告のみ
    inc = [0.0] * 10                                         # 次数9 (高容量=ノイズ過学習可能)
    accepted = 0
    for g in range(generations):
        cand = mutate(inc, rng)
        if accept(cand, inc, train, val):
            inc = cand
            accepted += 1
    return {
        "coefs": [round(c, 3) for c in inc],
        "train_mse": round(mse(inc, train), 4),
        "test_mse": round(mse(inc, test), 4),   # ← 真の汎化性能
        "accepted": accepted,
    }


def run_demo(seed: int = 1, generations: int = 1500, reps: int = 1000):
    def harness(c, i, tr, v):
        return harness_accepts(c, i, tr, v, reps=reps)
    naive = evolve(naive_accepts, generations=generations, seed=seed)
    harness_r = evolve(harness, generations=generations, seed=seed)
    # 真の最小達成可能 test_mse ≈ noise分散(=1.0)。それに近いほど汎化できている。
    return {"naive": naive, "harness": harness_r, "irreducible_test_mse": 1.0,
            "true_coefs": [1.0, -2.0, 0.5, 0.0, 0.0]}


CATASTROPHE_MULT = 5.0  # test 誤差が既約の何倍を超えたら「破滅的過学習」とみなすか


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def run_suite(seeds=(1, 2, 3, 4, 5, 6), generations: int = 800, reps: int = 300):
    """複数 seed で対比を集計。核心は中央値でなく**裾**:
    信頼淘汰は test 誤差を既約近くに**有界化**(破綻しない)、
    ナイーブ淘汰は seed 次第で**破滅的に過学習**(=1y3mループが時々大火傷した正体)。"""
    per_seed = []
    for s in seeds:
        r = run_demo(seed=s, generations=generations, reps=reps)
        per_seed.append({"seed": s,
                         "harness_test": r["harness"]["test_mse"],
                         "naive_test": r["naive"]["test_mse"],
                         "naive_train": r["naive"]["train_mse"],
                         "harness_train": r["harness"]["train_mse"]})
    h = [p["harness_test"] for p in per_seed]
    n = [p["naive_test"] for p in per_seed]
    irr = 1.0
    cat = CATASTROPHE_MULT * irr  # 破滅的過学習 = test 誤差が既約の CATASTROPHE_MULT 倍超
    return {
        "per_seed": per_seed,
        "irreducible_test_mse": irr,
        "catastrophe_threshold": cat,
        "harness": {"median_test": round(_median(h), 3), "max_test": round(max(h), 3)},
        "naive": {"median_test": round(_median(n), 3), "max_test": round(max(n), 3)},
        "naive_catastrophes": sum(1 for v in n if v > cat),
        "harness_catastrophes": sum(1 for v in h if v > cat),
    }


if __name__ == "__main__":
    import json
    r = run_suite()
    print(json.dumps(r, indent=2, ensure_ascii=False))
    print("\n--- 解釈 ---")
    print(f"既約 test_mse≈{r['irreducible_test_mse']} (ノイズ分散; これ未満は不可能)")
    print("seed別 test_mse (低いほど汎化):")
    for p in r["per_seed"]:
        flag = "  ← ナイーブ破滅" if p["naive_test"] > r["catastrophe_threshold"] else ""
        print(f"  seed{p['seed']}: 信頼淘汰={p['harness_test']:6.2f}  "
              f"ナイーブ={p['naive_test']:7.2f}{flag}")
    h, n = r["harness"], r["naive"]
    print(f"\n信頼淘汰  : 中央{h['median_test']} / 最悪{h['max_test']} / 破滅{r['harness_catastrophes']}件"
          f" → 既約近くに有界 = 騙されず汎化")
    print(f"ナイーブ淘汰: 中央{n['median_test']} / 最悪{n['max_test']} / 破滅{r['naive_catastrophes']}件"
          f" → seed次第で破滅的過学習 = train だけ見る罠")
    print("\n核心: 平均では大差なくても、ナイーブは**裾で大火傷する**(予測不能)。"
          "信頼淘汰は下振れを断つ。= bot の 1y3m ループが時々破綻した構造の縮図。")

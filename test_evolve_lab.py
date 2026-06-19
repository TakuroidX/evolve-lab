"""POC回帰ガード: 「信頼淘汰は下振れを断ち汎化、ナイーブ淘汰は裾で破滅的過学習」を固定する。

核心は平均勝率でなく**裾(最悪ケース)**。決定論(seed固定)なので値は再現する。
suite は重い(bootstrap)ので module 一度だけ計算して共有する。"""
import evolve_lab as el

# 全テスト共有の集計 (1 度だけ計算)。light params で数十秒以内。
SUITE = el.run_suite(seeds=(1, 2, 3, 4, 5, 6), generations=800, reps=300)
IRR = SUITE["irreducible_test_mse"]


def test_harness_bounds_the_downside():
    # 信頼淘汰は全 seed で test 誤差を既約の 5 倍以内に有界化し、破滅(>5×)を 1 件も出さない
    assert SUITE["harness"]["max_test"] < 5.0 * IRR
    assert SUITE["harness_catastrophes"] == 0


def test_naive_has_catastrophic_tail():
    # ナイーブ淘汰は train だけ見るため、一部 seed で破滅的に過学習する(裾リスク)
    assert SUITE["naive_catastrophes"] >= 1
    # 最悪ケースは信頼淘汰の最悪ケースより大幅に悪い (裾の差こそ本質)
    assert SUITE["naive"]["max_test"] > 2.0 * SUITE["harness"]["max_test"]


def test_harness_central_tendency_near_irreducible():
    # 中央値でも信頼淘汰は既約の 3 倍以内 (真の勾配が在るので登れている)
    assert SUITE["harness"]["median_test"] < 3.0 * IRR


def test_naive_overfits_on_susceptible_seed():
    # train を既約近くまで下げるのに test は大きく悪化する seed が在る (過学習の定義)
    overfit_seeds = [p for p in SUITE["per_seed"]
                     if p["naive_train"] < 1.2 and p["naive_test"] > 4.0 * p["naive_train"]]
    assert overfit_seeds, "ナイーブが train<1.2 かつ test>4×train になる seed が存在すること"


def test_harness_rejects_overfit_candidate():
    # held-out(val) を改善しない候補は淘汰器が却下する (val 無視の過学習を弾く unit test)
    train = el.make_dataset(12, 1.0, 1)
    val = el.make_dataset(60, 1.0, 8)
    inc = [1.0, -2.0, 0.5, 0.0, 0.0, 0.0]      # ほぼ真の係数 (次数5 まで持つ)
    overfit = [c for c in inc[:-1]] + [5.0]     # 最高次項を暴れさせた候補 = val では改善しない
    assert el.harness_accepts(overfit, inc, train, val, reps=300) is False


def test_harness_accepts_a_genuine_improvement():
    # 逆に val でも本物の改善になる候補は採用する (淘汰器が単に保守的なだけでない確認)
    train = el.make_dataset(12, 1.0, 1)
    val = el.make_dataset(60, 1.0, 8)
    bad = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]        # 全ゼロ (信号を全く捉えていない)
    better = [1.0, -2.0, 0.5, 0.0, 0.0, 0.0]    # 真の係数 = 明確に改善
    assert el.harness_accepts(better, bad, train, val, reps=300) is True

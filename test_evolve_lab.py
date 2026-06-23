"""POC回帰ガード(2026-06-23 confound 自己訂正後の正直版)。

固定する事実は2軸を分けたもの:
  - **held-out 評価そのもの**が裾の破滅を断つ (plain も harness も破滅0)。train だけ見る naive は破滅する。
  - この単純 toy では bootstrap/OOS **ゲートの限界貢献は plain held-out を上回らない**
    (ゲートの優位は別ドメイン ab_select/btc_exit が担う)。
核心は平均勝率でなく**裾(最悪ケース)**。決定論(seed固定)なので値は再現する。
suite は重い(bootstrap)ので module 一度だけ計算して共有する。"""
import evolve_lab as el

# 全テスト共有の集計 (1 度だけ計算)。light params で数十秒以内。
SUITE = el.run_suite(seeds=(1, 2, 3, 4, 5, 6), generations=800, reps=300)
IRR = SUITE["irreducible_test_mse"]


def test_holdout_bounds_the_downside():
    # held-out を使う 2 ルール(plain/harness)はどちらも全 seed で破滅(>5×既約)を 1 件も出さない。
    # = 裾を断つのは held-out 評価そのもの (ゲートに固有でない)。
    assert SUITE["plain"]["catastrophes"] == 0
    assert SUITE["harness"]["catastrophes"] == 0
    assert SUITE["plain"]["max_test"] < 5.0 * IRR
    assert SUITE["harness"]["max_test"] < 5.0 * IRR


def test_naive_train_only_has_catastrophic_tail():
    # train だけ見る naive (=1y3mループ型) は一部 seed で破滅的に過学習する(裾リスク)。
    assert SUITE["naive"]["catastrophes"] >= 1
    # 最悪ケースは held-out 選別(plain)の最悪より大幅に悪い = 「held-out を使うこと」が裾を断つ本質。
    assert SUITE["naive"]["max_test"] > 2.0 * SUITE["plain"]["max_test"]


def test_gate_does_not_beat_plain_holdout_in_this_toy():
    # 正直な confound 訂正の核: この単純 toy では bootstrap/OOS ゲートは plain held-out を上回らない。
    # (plain は破滅0かつ中央値も harness 以下) → 裾を断つ功績はゲートでなく held-out データにある。
    # ゲートが plain に勝つのは多重比較/regime偏り/打ち切り設定 = ab_select.py / btc_exit.py が実証。
    assert SUITE["plain"]["catastrophes"] == 0
    assert SUITE["plain"]["median_test"] <= SUITE["harness"]["median_test"]


def test_holdout_central_tendency_near_irreducible():
    # 中央値でも held-out 選別は既約の 3 倍以内 (真の勾配が在るので登れている)。
    assert SUITE["plain"]["median_test"] < 3.0 * IRR
    assert SUITE["harness"]["median_test"] < 3.0 * IRR


def test_naive_overfits_on_susceptible_seed():
    # train を既約近くまで下げるのに test は大きく悪化する seed が在る (過学習の定義)。
    overfit_seeds = [p for p in SUITE["per_seed"]
                     if p["naive_train"] < 1.2 and p["naive_test"] > 4.0 * p["naive_train"]]
    assert overfit_seeds, "ナイーブが train<1.2 かつ test>4×train になる seed が存在すること"


def test_harness_rejects_overfit_candidate():
    # held-out(val) を改善しない候補は淘汰器ゲートが却下する (val 無視の過学習を弾く unit test)。
    train = el.make_dataset(12, 1.0, 1)
    val = el.make_dataset(60, 1.0, 8)
    inc = [1.0, -2.0, 0.5, 0.0, 0.0, 0.0]      # ほぼ真の係数 (次数5 まで持つ)
    overfit = [c for c in inc[:-1]] + [5.0]     # 最高次項を暴れさせた候補 = val では改善しない
    assert el.harness_accepts(overfit, inc, train, val, reps=300) is False


def test_harness_accepts_a_genuine_improvement():
    # 逆に val でも本物の改善になる候補は採用する (淘汰器が単に保守的なだけでない確認)。
    train = el.make_dataset(12, 1.0, 1)
    val = el.make_dataset(60, 1.0, 8)
    bad = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]        # 全ゼロ (信号を全く捉えていない)
    better = [1.0, -2.0, 0.5, 0.0, 0.0, 0.0]    # 真の係数 = 明確に改善
    assert el.harness_accepts(better, bad, train, val, reps=300) is True

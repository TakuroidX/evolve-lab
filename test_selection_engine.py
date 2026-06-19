"""selection_engine の回帰ガード: 汎用ゲートの合否 + エンジンが勾配ドメインで汎化することを固定。"""
import evolve_lab as el
import selection_engine as se
import domains


def test_bootstrap_passes_real_improvement():
    dom = domains.symbolic_regression(seed=1)
    bad = [0.0] * 6
    good = [1.0, -2.0, 0.5, 0.0, 0.0, 0.0]  # ほぼ真の係数
    assert se.gate_bootstrap(dom, good, bad, reps=500).status == "pass"


def test_bootstrap_rejects_no_improvement():
    dom = domains.symbolic_regression(seed=1)
    inc = [1.0, -2.0, 0.5, 0.0, 0.0, 0.0]
    assert se.gate_bootstrap(dom, inc[:], inc, reps=500).status == "fail"  # 同一=改善なし


def test_oos_pass_on_consistent_improvement():
    dom = domains.symbolic_regression(seed=1)
    assert se.gate_oos(dom, [1.0, -2.0, 0.5, 0.0, 0.0, 0.0], [0.0] * 6, min_n=20).status == "pass"


def test_regime_insufficient_without_slice_key():
    dom = domains.symbolic_regression(seed=1, with_regime=False)
    assert se.gate_regime(dom, [1.0, -2.0, 0.5, 0.0, 0.0, 0.0], [0.0] * 6).status == "insufficient"


def test_regime_excludes_unknown_slice():
    # 40件 UNKNOWN + 20件 "A"。UNKNOWN を除外すると real slice は1個 → insufficient
    data = [("UNKNOWN" if i < 40 else "A", float(i)) for i in range(60)]
    dom = {"evaluate": lambda c, s: s[1] * c, "data": data,
           "slice_key": lambda s: s[0], "ordered_key": None, "variation": None}
    assert se.gate_regime(dom, 2.0, 1.0, min_n=20).status == "insufficient"


def test_select_full_scorecard_pass():
    dom = domains.symbolic_regression(seed=1, with_regime=True)
    v = se.select(dom, [1.0, -2.0, 0.5, 0.0, 0.0, 0.0], [0.0] * 6,
                  gates=[lambda d, c, i: se.gate_bootstrap(d, c, i, reps=500),
                         se.gate_oos, se.gate_regime])
    assert v.overall == "PASS"


def test_engine_evolves_and_generalizes():
    # 同じエンジンが勾配ドメインで [0]*6 から登り、未使用 test で汎化する (= evolve-lab 結果の再現)
    dom = domains.symbolic_regression(seed=1)
    test = el.make_dataset(400, 1.0, 1 + 9999)
    res = se.evolve(dom, [0.0] * 6, generations=300, seed=1,
                    accept_gates=[lambda d, c, i: se.gate_bootstrap(d, c, i, reps=400)])
    assert res["accepted"] > 0
    assert el.mse(res["incumbent"], test) < 4.0  # 既約≈1.0 の数倍以内に汎化

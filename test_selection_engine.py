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


def test_censoring_insufficient_without_key():
    # censored_key を持たないドメイン (記号回帰) では gate_censoring は insufficient (巻き込まない)
    dom = domains.symbolic_regression(seed=1)
    assert se.gate_censoring(dom, [1.0, -2.0, 0.5, 0.0, 0.0, 0.0], [0.0] * 6).status == "insufficient"


def test_censoring_insufficient_when_mostly_unobservable():
    # 18/20 が打ち切り → 観測可能 2 < min_observed → insufficient (判定保留=ship しない)
    dom = {"data": list(range(20)), "evaluate": lambda c, s: 1.0 if c == "B" else 0.0,
           "censored_key": lambda c, s: s < 18}
    assert se.gate_censoring(dom, "B", "A", min_observed=10).status == "insufficient"


def test_censoring_fail_when_improvement_is_censoring_driven():
    # 打ち切り部では candidate +1 だが観測可能部では -0.5 → 全体+ でも観測可能部で消失 → fail
    dom = {"data": list(range(20)),
           "evaluate": lambda c, s: (1.0 if s < 10 else -0.5) if c == "B" else 0.0,
           "censored_key": lambda c, s: s < 10}
    r = se.gate_censoring(dom, "B", "A", min_observed=10)
    assert r.status == "fail" and r.numbers["mean_all"] > 0 and r.numbers["mean_observed"] <= 0


def test_censoring_pass_when_improvement_holds_observed():
    # 改善が観測可能部でも残る (どこでも +0.5・打ち切り 20%) → pass
    dom = {"data": list(range(20)), "evaluate": lambda c, s: 0.5 if c == "B" else 0.0,
           "censored_key": lambda c, s: s < 4}
    assert se.gate_censoring(dom, "B", "A", min_observed=10).status == "pass"


def test_engine_evolves_and_generalizes():
    # 同じエンジンが勾配ドメインで [0]*6 から登り、未使用 test で汎化する (= evolve-lab 結果の再現)
    dom = domains.symbolic_regression(seed=1)
    test = el.make_dataset(400, 1.0, 1 + 9999)
    res = se.evolve(dom, [0.0] * 6, generations=300, seed=1,
                    accept_gates=[lambda d, c, i: se.gate_bootstrap(d, c, i, reps=400)])
    assert res["accepted"] > 0
    assert el.mse(res["incumbent"], test) < 4.0  # 既約≈1.0 の数倍以内に汎化

"""ab_select ドメインの回帰ガード (決定論・API 不要)。

主張は2層に分けて誇張を避ける (検収レビュー #1):
  - **構造的・seed頑健**: peeking/concentrated は naive が必ず出荷する偽陽性で、別々のゲート
    (oos / regime) が必ず捕まえる。genuine は真陽性で全ゲート PASS。
  - **確率的 (noise)**: 真の効果≈0 は naive が「+に転んだ時だけ」出荷し (seed依存)、bootstrap が
    ノイズフロアで却下する (大多数)。「noise も必ず4/4」とは主張しない=ノイズの本質。
"""
import ab_select as ab
import domains
import selection_engine as se

GATES = [lambda d, c, i: se.gate_bootstrap(d, c, i, reps=500),
         lambda d, c, i: se.gate_oos(d, c, i, min_n=20),
         lambda d, c, i: se.gate_regime(d, c, i, min_n=20)]
ROBUST_SEEDS = range(1, 9)  # 構造的主張は複数 seed で確認 (cherry-pick でない)


def _verdict(scenario, seed=1):
    dom = domains.ab_select(scenario, n=120, seed=seed)
    v = se.select(dom, "b", "a", gates=GATES)
    return dom, v, {r.name: r.status for r in v.gates}


# --- 構造的・seed頑健な主張 ---
def test_naive_ships_structural_winners_every_seed():
    # genuine(真陽性) と peeking/concentrated(偽陽性) は全 seed で naive が B を出荷
    for sc in ("genuine", "peeking", "concentrated"):
        for s in ROBUST_SEEDS:
            assert ab.naive_winner(domains.ab_select(sc, n=120, seed=s))["ships_b"] is True


def test_genuine_passes_all_gates_every_seed():
    for s in ROBUST_SEEDS:
        _, v, g = _verdict("genuine", s)
        assert v.overall == "PASS"
        assert g == {"bootstrap": "pass", "oos": "pass", "regime": "pass"}


def test_peeking_caught_by_oos_every_seed():
    # 全体平均は勝ちに見える (bootstrap pass) が、期間ブロックで符号反転 → oos が捕捉
    for s in ROBUST_SEEDS:
        _, v, g = _verdict("peeking", s)
        assert v.overall == "FAIL"
        assert g["bootstrap"] == "pass"   # 集計は騙される
        assert g["oos"] == "fail"         # block(週順)間で符号反転


def test_concentrated_caught_by_regime_every_seed():
    # 全体平均+ かつ期間一様 (bootstrap/oos pass) だが 1セグメント偏重 → regime が捕捉
    for s in ROBUST_SEEDS:
        _, v, g = _verdict("concentrated", s)
        assert v.overall == "FAIL"
        assert g["bootstrap"] == "pass"
        assert g["oos"] == "pass"
        assert g["regime"] == "fail"


def test_engine_rejects_structural_false_positives_every_seed():
    # 構造的偽陽性 2件は全 seed で FAIL (naive は全部出荷していた)
    for sc in ("peeking", "concentrated"):
        for s in ROBUST_SEEDS:
            assert _verdict(sc, s)[1].overall == "FAIL"


# --- 確率的主張 (noise): 誇張しない ---
def test_noise_is_probabilistic_not_always_shipped():
    # noise の naive 出荷は seed 依存 (= 必ずではない)。これを明示的に固定し「4/4 構造的」誤主張を防ぐ。
    ships = [ab.naive_winner(domains.ab_select("noise", n=120, seed=s))["ships_b"]
             for s in range(1, 21)]
    assert any(ships) and not all(ships)  # 出る seed も出ない seed もある = 確率的


def test_noise_rejected_by_bootstrap_strong_majority():
    # bootstrap はノイズフロアで noise を大多数 却下する (頑健だが 100% ではない=正直に)
    rejected = sum(se.gate_bootstrap(domains.ab_select("noise", n=120, seed=s), "b", "a",
                                     reps=400).status == "fail" for s in range(1, 21))
    assert rejected >= 17  # >=85% (実測 19/20)


def test_noise_caught_at_pinned_seed1():
    _, v, g = _verdict("noise", 1)  # seed=1 では naive 出荷(+0.69)・bootstrap 却下
    assert ab.naive_winner(domains.ab_select("noise", n=120, seed=1))["ships_b"] is True
    assert g["bootstrap"] == "fail" and v.overall == "FAIL"


# --- ガード (検収 #2) ---
def test_min_n_floor_guard():
    raised = False
    try:
        domains.ab_select("genuine", n=59, seed=1)
    except ValueError:
        raised = True
    assert raised

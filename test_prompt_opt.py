"""prompt_opt ドメインの回帰ガード (決定論 fake・API 不要)。

検証する主張:
  1. 同じ淘汰エンジンが「本物の勾配のある」プロンプト最適化ドメインで登る (汎化は held-out で確認)。
  2. 淘汰器が自己欺瞞を弾く: 効かない変更 (どの sample も要らないマーカー) は bootstrap が却下、
     1 regime でしか効かない変更は regime ゲートが却下。
  3. キャッシュが (prompt,input) を重複計算しない (=実 API コストを抑える) / 上限で暴走しない。
"""
import random

import domains
import prompt_opt as po
import selection_engine as se

ALL_USEFUL = f"{po.BASE_PROMPT} only concise number"  # 全 useful マーカー入り = 満点プロンプト


def test_domain_shape():
    dom = domains.prompt_opt(n=12)
    for k in ("variation", "evaluate", "data", "ordered_key", "slice_key"):
        assert k in dom
    assert len(dom["data"]) == 12
    assert callable(dom["evaluate"]) and callable(dom["variation"])


def test_fake_model_has_real_gradient():
    # 全マーカー入りプロンプトは base より全 sample で高得点 = 勾配が実在
    dom = domains.prompt_opt(n=24, seed=1)
    assert po.mean_score(dom, po.BASE_PROMPT) == 0.0
    assert po.mean_score(dom, ALL_USEFUL) == 1.0


def test_bootstrap_accepts_needed_marker():
    # base に "concise" 追加 → "add" regime の sample が flip → 平均改善が本物 → pass
    dom = domains.prompt_opt(n=24, seed=1)
    cand = f"{po.BASE_PROMPT} concise"
    assert se.gate_bootstrap(dom, cand, po.BASE_PROMPT, reps=500).status == "pass"


def test_bootstrap_rejects_useless_marker():
    # どの sample も要らない useless マーカーを足しても改善ゼロ → ノイズと区別不能 → pass しない
    dom = domains.prompt_opt(n=24, seed=1)
    cand = f"{po.BASE_PROMPT} {po.USELESS_MARKER}"
    assert se.gate_bootstrap(dom, cand, po.BASE_PROMPT, reps=500).status != "pass"


def test_regime_gate_rejects_single_regime_win():
    # "concise" は "add" だけ助け "extract"/"format" は不変 → 全 slice で+でない → regime fail
    dom = domains.prompt_opt(n=30, seed=1)
    cand = f"{po.BASE_PROMPT} concise"
    assert se.gate_regime(dom, cand, po.BASE_PROMPT, min_n=8).status == "fail"


def test_regime_gate_passes_all_regime_win():
    # 全 useful マーカー → 全 regime で改善 → regime pass (全 slice で+)
    dom = domains.prompt_opt(n=30, seed=1)
    assert se.gate_regime(dom, ALL_USEFUL, po.BASE_PROMPT, min_n=8).status == "pass"


def test_engine_climbs_and_generalizes():
    # bootstrap-only の軽い受理ゲートで evolve → 受理が起き、未使用 held-out で base より汎化改善
    dom = domains.prompt_opt(n=24, seed=1)
    res = se.evolve(dom, po.BASE_PROMPT, generations=200, seed=3,
                    accept_gates=[lambda d, c, i: se.gate_bootstrap(d, c, i, reps=300)])
    assert res["accepted"] > 0
    held_out = domains.prompt_opt(n=30, seed=999)  # 別 seed = 選別に未使用の評価集合
    assert po.mean_score(held_out, res["incumbent"]) > po.mean_score(held_out, po.BASE_PROMPT)


def test_full_scorecard_pass_for_evolved_prompt():
    # 全 useful マーカー版は全ゲート (bootstrap+oos+regime) を通る = ship 候補 (但し PASS≠ship; Lesson-6)
    dom = domains.prompt_opt(n=30, seed=1)
    v = se.select(dom, ALL_USEFUL, po.BASE_PROMPT,
                  gates=[lambda d, c, i: se.gate_bootstrap(d, c, i, reps=500),
                         lambda d, c, i: se.gate_oos(d, c, i, min_n=8),
                         lambda d, c, i: se.gate_regime(d, c, i, min_n=8)])
    assert v.overall == "PASS"


def test_cache_avoids_recompute():
    # 同一 (prompt,input) の重複評価で unique 呼び出しが増えない (=実 API なら課金されない)
    dom = domains.prompt_opt(n=10, seed=1)
    cache = dom["_cache"]
    for _ in range(5):
        po.mean_score(dom, ALL_USEFUL)
    assert cache.calls == 10  # 10 sample を1回ずつだけ計算 (5周しても増えない)


def test_cache_max_calls_guard():
    dom = domains.prompt_opt(n=50, seed=1, max_calls=5)
    raised = False
    try:
        po.mean_score(dom, "some unique prompt xyz")  # 50 sample > 上限5
    except RuntimeError:
        raised = True
    assert raised


def test_max_rewrites_guard():
    # 変異(rewrite)も上限で縛れる (review #2: 評価キャッシュは rewrite コストを覆わない)
    dom = domains.prompt_opt(n=12, seed=1, max_rewrites=3)
    rng = random.Random(0)
    raised = False
    try:
        for _ in range(10):  # 上限3 を超えて変異を要求
            dom["variation"](po.BASE_PROMPT, rng)
    except RuntimeError:
        raised = True
    assert raised
    assert dom["_rewrites"].calls == 3

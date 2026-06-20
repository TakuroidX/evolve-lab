"""demo_prompt_opt.py — prompt_opt ドメインを**実 Anthropic API**で回すデモ (コスト発生)。

テスト/CI は決定論 fake (test_prompt_opt.py)。これは「本物の LLM でも同じエンジンが登り、
淘汰器が ship 候補を全ゲートで検証する」ことを見せる実走スクリプト。

実行 (鍵を export してから・コスト発生):
    export ANTHROPIC_API_KEY=...           # リポジトリには鍵を置かない
    python3 demo_prompt_opt.py             # 既定: 小さく回す (~数十 API 呼び出し・haiku)

コストガード: 安価モデル + 評価キャッシュ + max_calls 上限 + 小 n/世代。実 API 呼び出し数を最後に表示。
"""
from __future__ import annotations

import sys

import domains
import prompt_opt as po
import selection_engine as se


def main(n: int = 12, generations: int = 8, max_calls: int = 400,
         max_rewrites: int = 40, seed: int = 3) -> int:
    # ⚠️ n>=10 必須: accept gate=gate_bootstrap は n<10 で insufficient → 受理ゼロ=登れず API 浪費 (review #1)
    try:
        from anthropic_backend import make_model_fn, make_rewrite_fn
        model_fn = make_model_fn()
        rewrite_fn = make_rewrite_fn()
    except RuntimeError as e:
        print(f"[skip] {e}")
        return 0

    dom = domains.prompt_opt(model_fn=model_fn, rewrite_fn=rewrite_fn, n=n, seed=seed,
                             max_calls=max_calls, max_rewrites=max_rewrites)
    cache = dom["_cache"]
    rewrites = dom["_rewrites"]

    print(f"=== prompt_opt 実 API デモ (n={n}, generations={generations}, model=haiku) ===")
    print(f"出発 prompt : {po.BASE_PROMPT!r}")
    print(f"出発 score  : {po.mean_score(dom, po.BASE_PROMPT):.3f}")

    res = se.evolve(dom, po.BASE_PROMPT, generations=generations, seed=seed,
                    accept_gates=[lambda d, c, i: se.gate_bootstrap(d, c, i, reps=300)])
    evolved = res["incumbent"]
    print(f"\n進化後 prompt: {evolved!r}")
    print(f"進化後 score : {po.mean_score(dom, evolved):.3f}  (受理 {res['accepted']} 回)")

    v = se.select(dom, evolved, po.BASE_PROMPT,
                  gates=[lambda d, c, i: se.gate_bootstrap(d, c, i, reps=500),
                         lambda d, c, i: se.gate_oos(d, c, i, min_n=2),
                         lambda d, c, i: se.gate_regime(d, c, i, min_n=2)])
    print(f"\n最終スコアカード (進化後 vs 出発): {v.overall}")
    for g in v.gates:
        print(f"  - {g.name:9s}: {g.status:12s} {g.reason}")
    print("  ※ PASS≠ship (Lesson-6): 最終判断は人/critic。これは計器の出力。")

    print(f"\n--- コスト (実 API 呼び出し) ---")
    print(f"  model_fn unique 呼び出し: {cache.calls}  /  rewrite 呼び出し: {rewrites.calls}")
    print(f"  合計 ~{cache.calls + rewrites.calls} 呼び出し (haiku・キャッシュで重複は課金なし)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

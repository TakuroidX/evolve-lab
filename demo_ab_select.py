"""demo_ab_select.py — A/B 選別の規律デモ (決定論・API 不要・完全無料)。

ナイーブ選別 (平均が高い方を採用) vs 同じ淘汰器のスコアカードを4シナリオで対比表示。
偽の勝ち3つ (noise/peeking/concentrated) を、別々のゲートが捕まえる様子を見せる。

    python3 demo_ab_select.py
"""
from __future__ import annotations

import ab_select as ab
import domains
import selection_engine as se

GATES = [lambda d, c, i: se.gate_bootstrap(d, c, i, reps=800),
         lambda d, c, i: se.gate_oos(d, c, i, min_n=20),
         lambda d, c, i: se.gate_regime(d, c, i, min_n=20)]

CAUGHT_BY = {"genuine": "— (真陽性)", "noise": "bootstrap (ノイズフロア)",
             "peeking": "oos (期間ブロックで符号反転)", "concentrated": "regime (1セグメント偏重)"}


def main() -> int:
    print("=== A/B 選別: ナイーブ平均比較 vs 信頼できる淘汰 (決定論・無料・seed=1) ===\n")
    print(f"{'シナリオ':14s} {'ナイーブ':18s} {'エンジン':6s} {'boot':5s} {'oos':5s} {'reg':5s}  捕捉ゲート")
    print("-" * 90)
    for sc in ab.SCENARIOS:
        dom = domains.ab_select(sc, n=120, seed=1)
        nv = ab.naive_winner(dom)
        v = se.select(dom, "b", "a", gates=GATES)
        g = {r.name: r.status for r in v.gates}
        naive = f"出荷B (Δ{nv['mean_diff']:+.2f})" if nv["ships_b"] else "据置A"
        print(f"{sc:14s} {naive:18s} {v.overall:6s} {g['bootstrap']:5s} {g['oos']:5s} "
              f"{g['regime']:5s}  {CAUGHT_BY[sc]}")
    print("-" * 90)

    # 構造的偽陽性 (peeking/concentrated) は seed 横断で頑健・noise は確率的 — 誇張しない
    print("\n[構造的偽陽性は seed 横断で頑健]")
    for sc in ("peeking", "concentrated"):
        ships = sum(ab.naive_winner(domains.ab_select(sc, 120, s))["ships_b"] for s in range(1, 21))
        fails = sum(se.select(domains.ab_select(sc, 120, s), "b", "a", gates=GATES).overall == "FAIL"
                    for s in range(1, 21))
        print(f"  {sc:13s}: ナイーブ出荷 {ships}/20 seed · エンジン却下 {fails}/20 seed")
    sh = sum(ab.naive_winner(domains.ab_select("noise", 120, s))["ships_b"] for s in range(1, 21))
    bf = sum(se.gate_bootstrap(domains.ab_select("noise", 120, s), "b", "a", reps=400).status
             == "fail" for s in range(1, 21))
    print(f"\n[noise は確率的 (=ノイズの本質・「必ず4/4」とは言わない)]")
    print(f"  noise        : ナイーブ出荷 {sh}/20 seed (運次第) · bootstrap 却下 {bf}/20 seed")

    print("\n→ ナイーブ A/B は『平均が高い方』を出荷する。peeking/concentrated は **必ず** 偽出荷し、")
    print("  淘汰器が別々の構造ゲートで **必ず** 止める。noise も大多数で止める (確率的)。")
    print("※ PASS≠ship (Lesson-6): 最終判断は人/critic。これは計器の出力。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

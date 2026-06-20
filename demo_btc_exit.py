"""demo_btc_exit.py — 淘汰エンジンが BTC bot 自身の出口候補を裁くデモ (read-only)。

  python3 demo_btc_exit.py                                   # 合成 fixture (公開・無料)
  python3 demo_btc_exit.py --paths '/abs/logs/position_path_*.jsonl'  # bot の実 path (ローカル)

実 path を渡すと bot の実 SIM 出口を read-only で評価する (取引データは公開リポに焼かない方針)。
incumbent(現行出口) vs 代表3候補を 1個ずつ判定 + censoring を surface (緩和方向は veto)。
"""
from __future__ import annotations

import argparse

import btc_exit as bx
import domains
import selection_engine as se

GATES = [lambda d, c, i: se.gate_bootstrap(d, c, i, reps=800),
         lambda d, c, i: se.gate_oos(d, c, i, min_n=20),
         lambda d, c, i: se.gate_regime(d, c, i, min_n=20)]
DESC = {"tight_ts": "早利確 (TS を狭く)", "tight_sl": "早損切り (SL 0.6→0.4)",
        "loose_ts": "利伸ばし (TS 幅↑ = 緩和方向)"}
LOOSEN = {"loose_ts"}  # 緩和方向 = censoring veto 対象


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default=None, help="bot の position_path_*.jsonl glob (未指定=合成)")
    ap.add_argument("--regime", default="trend", choices=["trend", "side", "vol"])
    args = ap.parse_args()

    dom = domains.btc_exit(path_glob=args.paths, regime=args.regime)
    src = f"実 path ({args.paths})" if args.paths else "合成 fixture (公開・無料)"
    n = len(dom["data"])
    print(f"=== BTC bot 出口を淘汰器で裁く ({src}, n={n} trade, regime={args.regime}) ===")
    if n == 0:
        print("path ゼロ。--paths の glob を確認。")
        return 0
    slices = {}
    for p in dom["data"]:
        slices[dom["slice_key"](p)] = slices.get(dom["slice_key"](p), 0) + 1
    print(f"regime slices: {slices}\n")
    print(f"{'候補':22s} {'判定':6s} {'boot':5s} {'oos':5s} {'reg':5s} {'censored':9s} 備考")
    print("-" * 84)
    for key in ("tight_ts", "tight_sl", "loose_ts"):
        cand = bx.CANDIDATES[key]
        v = se.select(dom, cand, bx.INCUMBENT, gates=GATES)
        g = {r.name: r.status for r in v.gates}
        cr = bx.censored_rate(dom["data"], cand)
        note = ""
        if key in LOOSEN and cr > 30:
            note = f"⚠ 緩和×censored {cr}% → 結論 veto (観測不能)"
        print(f"{DESC[key]:22s} {v.overall:6s} {g['bootstrap']:5s} {g['oos']:5s} "
              f"{g['regime']:5s} {cr:7.1f}%  {note}")
    print("-" * 84)
    print("淘汰器は ranking せず 1候補ずつ判定 (Lesson-5)。PASS≠ship (Lesson-6: 最終は人/critic)。")
    print("緩和方向は右側打ち切りで観測不能 → 汎用ゲートが pass でも censoring veto が優先 (bot §I)。")

    # censoring veto が「実仕事をする」実演: 3ゲートが PASS でも 100% censored なら veto が却下
    trap = bx.make_censoring_trap_paths(n=90, seed=1)
    tdom = bx.build_btc_exit_domain(trap, regime_key=bx.trend_phase)
    tv = se.select(tdom, bx.CANDIDATES["loose_ts"], bx.INCUMBENT, gates=GATES)
    tg = {r.name: r.status for r in tv.gates}
    tcr = bx.censored_rate(trap, bx.CANDIDATES["loose_ts"])
    print(f"\n[censoring veto の実演 — 押し目→高値close の trap fixture, n={len(trap)}]")
    print(f"  利伸ばし(loose): 3ゲート判定={tv.overall} (boot={tg['bootstrap']}/oos={tg['oos']}/reg={tg['regime']})"
          f" だが censored={tcr}%")
    print(f"  → 汎用ゲートは『本物の改善』に見せられる。**censoring veto が PASS を却下** "
          f"(実 close から先は観測不能=緩和の真の効果は測れない)。これが gate 単体で不十分な理由。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

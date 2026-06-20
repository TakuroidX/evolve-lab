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

# 4ゲート: bootstrap/oos/regime + censoring (step6 で汎用ゲート化した非対称 veto)
GATES = [lambda d, c, i: se.gate_bootstrap(d, c, i, reps=800),
         lambda d, c, i: se.gate_oos(d, c, i, min_n=20),
         lambda d, c, i: se.gate_regime(d, c, i, min_n=20),
         lambda d, c, i: se.gate_censoring(d, c, i)]
DESC = {"tight_ts": "早利確 (TS を狭く)", "tight_sl": "早損切り (SL 0.6→0.4)",
        "loose_ts": "利伸ばし (TS 幅↑ = 緩和方向)"}


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
    print(f"{'候補':22s} {'判定':6s} {'boot':5s} {'oos':5s} {'reg':5s} {'cens':5s} (censored%)")
    print("-" * 80)
    for key in ("tight_ts", "tight_sl", "loose_ts"):
        cand = bx.CANDIDATES[key]
        v = se.select(dom, cand, bx.INCUMBENT, gates=GATES)
        g = {r.name: r.status for r in v.gates}
        cr = bx.censored_rate(dom["data"], cand)
        print(f"{DESC[key]:22s} {v.overall:6s} {g['bootstrap']:5s} {g['oos']:5s} "
              f"{g['regime']:5s} {g['censoring']:5s} ({cr:.1f}%)")
    print("-" * 80)
    print("淘汰器は ranking せず 1候補ずつ判定 (Lesson-5)。PASS≠ship (Lesson-6: 最終は人/critic)。")
    print("censoring は 4本目の汎用ゲート (step6): 打ち切り「率」でなく**観測可能部で改善が残るか**を見る")
    print("  = tighten(観測窓内)は高打ち切りでも残れば pass / loosen(close超え)は観測不能で降格 = 非対称。")

    # censoring ゲートが「実仕事をする」実演: 他3ゲートが PASS でも観測不能なら総合を降格させる
    trap = bx.make_censoring_trap_paths(n=90, seed=1)
    tdom = bx.build_btc_exit_domain(trap, regime_key=bx.trend_phase)
    tv = se.select(tdom, bx.CANDIDATES["loose_ts"], bx.INCUMBENT, gates=GATES)
    tg = {r.name: r.status for r in tv.gates}
    print(f"\n[censoring ゲートの実演 — 押し目→高値close の trap fixture, n={len(trap)}]")
    print(f"  利伸ばし(loose): 総合={tv.overall} | boot={tg['bootstrap']} oos={tg['oos']} "
          f"reg={tg['regime']} → **censoring={tg['censoring']}** (観測可能 sample ゼロ)")
    print(f"  → bootstrap/oos/regime は『本物の改善』に騙される。だが loose は実 close から先が観測不能 →")
    print(f"     **censoring ゲートが総合を PASS から降格 (=ship しない)**。これが汎用化した非対称 veto。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

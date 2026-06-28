"""demo_lookahead_leak.py — a standalone, dependency-free reproduction of the look-ahead
leak that inflated a real trading model's headline metric from a true ~0.52 (coin-flip) to ~0.71.

THE BUG (real, found in my own model on 2026-06-28):
  Multi-timeframe features were merged onto the 5-minute decision bar with
  `pandas.merge_asof(direction="backward")` keyed on each bar's *open* timestamp.
  At a 5-min decision point t, that attaches the higher-timeframe (e.g. 1h) bar that is
  still FORMING (open <= t < close). In bulk/historical processing that forming bar already
  holds its FINAL close — a price up to ~1h in the *future*. So the feature silently encoded
  the future. Live, the forming bar is only partial, so the leak vanishes and the model is
  revealed as ~coin-flip. That single bug was the difference between "offline 0.71" and
  "live 0.50" for fifteen months.

THIS SCRIPT proves the mechanism on *pure random-walk* synthetic data — i.e. data with
**zero real predictability**. Any AUC above 0.5 here is the leak and nothing else:
  - leaky HTF feature  (attaches the forming 1h bar, open<=t)  -> forward-AUC ~0.7
  - causal HTF feature (attaches the last *completed* 1h bar)   -> forward-AUC ~0.5

Run:  python3 demo_lookahead_leak.py            (deterministic, stdlib only, <1s)
"""
from __future__ import annotations
import random

BARS_PER_HOUR = 12          # 5-min bars per 1h bar
HORIZON = 2                 # label = +10min direction (2 x 5-min bars)


def random_walk(n: int, seed: int = 0) -> list[float]:
    """A pure geometric-ish random walk: NO predictable structure exists in it."""
    rng = random.Random(seed)
    p = [10_000_000.0]
    for _ in range(n - 1):
        p.append(p[-1] * (1.0 + rng.gauss(0.0, 0.0008)))
    return p


def rank_auc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney rank AUC (ties averaged). Pure Python."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def build(prices: list[float]):
    """At each 5-min bar t, build a leaky vs a causal 1h feature, plus the +10min label."""
    leaky, causal, labels = [], [], []
    n = len(prices)
    for t in range(BARS_PER_HOUR, n - HORIZON):
        j = t // BARS_PER_HOUR                      # index of the 1h bar CONTAINING t (still forming at t)
        h_open, h_close = j * BARS_PER_HOUR, j * BARS_PER_HOUR + BARS_PER_HOUR - 1
        if h_close >= n:
            continue
        # LEAK: merge_asof(backward, open-time) attaches the forming bar j; in bulk data its
        #       close = prices[h_close] is in the FUTURE relative to t. We read that future.
        leaky.append(prices[h_close] - prices[h_open])
        # CAUSAL: only the last fully-COMPLETED 1h bar (j-1), entirely in the past.
        pj = j - 1
        causal.append(prices[pj * BARS_PER_HOUR + BARS_PER_HOUR - 1] - prices[pj * BARS_PER_HOUR])
        # label: did price rise over the next +10min?
        labels.append(1 if prices[t + HORIZON] > prices[t] else 0)
    return leaky, causal, labels


def run(seed: int = 0, n: int = 20_000) -> dict:
    prices = random_walk(n, seed)
    leaky, causal, labels = build(prices)
    a_leak = rank_auc(leaky, labels)
    a_causal = rank_auc(causal, labels)
    return {"n": len(labels), "leaky_auc": a_leak, "causal_auc": a_causal}


if __name__ == "__main__":
    r = run()
    print("=" * 64)
    print("LOOK-AHEAD LEAK DEMO  (pure random walk = zero real signal)")
    print("=" * 64)
    print(f"  samples                         : {r['n']}")
    print(f"  LEAKY  1h feature (forming bar) : forward-AUC = {r['leaky_auc']:.3f}   <- 'skill' that is 100% future-reading")
    print(f"  CAUSAL 1h feature (closed bar)  : forward-AUC = {r['causal_auc']:.3f}   <- the truth (coin-flip)")
    print("-" * 64)
    print("  The data has NO predictable structure. The only thing the 'leaky' feature")
    print("  predicts is the future it was accidentally allowed to see. This is exactly")
    print("  how a real model showed offline 0.71 while being ~0.50 live.")
    print("=" * 64)

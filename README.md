# evolve-lab

**A minimal, deterministic demonstration that the bottleneck in a self-improving loop is
*trustworthy selection*, not mutation — and that the same selection discipline that catches a
self-improving loop fooling itself can be made domain-agnostic and reused.**

Pure Python, zero dependencies, fully deterministic (fixed seeds). Independent of the trading
bot it was distilled from. For the longer narrative, see [`STORY.md`](STORY.md). 日本語は [`README.ja.md`](README.ja.md)。

```bash
python3 evolve_lab.py     # POC: contrast trustworthy vs naive selection across seeds
python3 -m pytest -q      # regression guards (evolve_lab 6 + selection_engine 7 = 13 tests)
```

---

## TL;DR

Over ~1.5 years I built a self-improving algorithmic-trading bot. It never found a durable edge.
The honest, hard-won lesson was **not** "the model was too weak" — it was that **a self-improvement
loop's hardest problem is selecting real improvements from noise without deceiving itself**, and that
on a near-flat / non-stationary / censored fitness landscape, *more mutation just overfits faster*.

This repo isolates that lesson in two pieces:

1. **`evolve_lab.py`** — a symbolic-regression POC where the hidden signal is real (a true gradient
   exists). Run the *same* evolutionary loop with two selection rules and watch them diverge:
   - **Trustworthy selection** (accept only if the improvement is statistically real on held-out data)
     → generalizes, bounded near the irreducible error, **zero blow-ups**.
   - **Naive selection** (accept if training error drops — the "loss → tweak → looks better → new problem"
     loop) → **catastrophically overfits on some seeds (8–15× worse), unpredictably**.

2. **`selection_engine.py`** — the discipline extracted into one **domain-agnostic object**:
   variation × selection (a pass/fail scorecard of generic gates) × inheritance. Plug in a fitness
   function and data and it runs on any domain. The trading bot becomes just one (edgeless) domain it
   can evaluate; symbolic regression is the domain where it visibly climbs.

## The question it answers

When a self-improving system fails to improve, there are two suspects: **the method**
(variation × selection × inheritance) or **the domain** (is there a real gradient to climb?).
For the trading bot, the method looked plausible but the result was null — so which was it?

This POC isolates the variable. On a domain where a gradient provably exists, the same loop with
trustworthy selection **does** climb. So the bot's failure points to the *domain* — no exploitable
gradient in the inputs we could access — rather than the method. (Honest caveat: this shows the method
*works where signal exists*; it cannot prove that better data or a better implementation would never
have found an edge.) The corollary is the warning: with naive,
train-only selection, the loop doesn't just fail to climb — it **drifts into overfitting**, exactly
the "1.5-year loop" of loss → add a filter → short-term improvement → a new problem → repeat.

## Results (seeds 1–6, deterministic)

```
irreducible test MSE ≈ 1.0  (the noise variance; nothing can beat this)

  seed   trustworthy   naive
   1        1.35        8.98   ← naive blow-up
   2        1.92        1.49
   3        1.59       15.46   ← naive blow-up
   4        3.95        2.84
   5        2.75        1.93
   6        2.56        6.59   ← naive blow-up

trustworthy : median 2.24 / worst  3.95 / blow-ups 0   → bounded near irreducible, generalizes
naive       : median 4.72 / worst 15.46 / blow-ups 3   → catastrophic overfit on some seeds
```

**The point is the tail, not the mean.** On average the two are not far apart; the difference is that
naive selection **blows up 8–15× on a fraction of seeds, unpredictably**, while trustworthy selection
caps the downside. This mirrors how the bot's loop failed: not every time, but occasionally and
without warning.

> **Honest note:** this is *not* "trustworthy selection always wins" — on seeds 2 and 5 naive
> generalizes fine. The claim is narrow and exact: **trustworthy selection removes the catastrophic
> tail (0 blow-ups); naive selection carries it (3/6).** Not overstating the result is the entire point.

## The selection engine

`selection_engine.py` generalizes the bot's `fitness.py` harness into a reusable object. A *domain*
is just a dict: `variation`, `evaluate`, `data`, and optional `ordered_key` (for time-series OOS) /
`slice_key` (for regime robustness). Three generic, self-deception-resistant gates:

- **`gate_bootstrap`** — the per-sample improvement's 95% bootstrap CI must exclude 0 (above the noise floor).
- **`gate_oos`** — time-series k-fold: the improvement must hold its sign across every time block (no period-overfit).
- **`gate_regime`** — the improvement must not flip sign across regime slices (`"UNKNOWN"` slices are excluded, not silently bucketed).

`select()` returns a **scorecard** (PASS / HOLD / FAIL) — it does not rank candidates (ranking invites
the winner's curse). `evolve()` runs the loop with a configurable acceptance gate. Adding a new domain
(`domains.py`) is one function; that is what "one engine" means.

```bash
python3 -c "import evolve_lab as el, selection_engine as se, domains; \
  print(el.mse(se.evolve(domains.symbolic_regression(), [0.0]*6, 300, 1, \
  accept_gates=[lambda d,c,i: se.gate_bootstrap(d,c,i,reps=400)])['incumbent'], \
  el.make_dataset(400,1.0,10000)))"   # engine generalizes on a gradient domain → test_mse ≈ 1.7
```

## Honest positioning (what this is, and is **not**)

None of the individual techniques here are novel. They are well-established:

- **Backtest-overfitting statistics** — the Deflated Sharpe Ratio, Probability of Backtest Overfitting,
  and Combinatorial Purged Cross-Validation (Bailey & López de Prado) are the rigorous, superior forms
  of the bootstrap / OOS / contamination gates used here.
- **Self-improving agents** — the [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) (Sakana/UBC),
  [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (DeepMind),
  and the framing that a self-improvement loop is a multiple-testing procedure that can deceive itself.
- **LLM-driven evolution on trading specifically** —
  [MadEvolve](https://arxiv.org/abs/2605.23007) (Kvasiuk et al., May 2026) runs an LLM evolutionary loop
  on BTC OHLCV with chronological train/val/test and explicit *p-hacking* analysis — strikingly close in
  *approach*. The conclusions differ: MadEvolve reports positive results on its setup, whereas this
  project's bot found no durable edge under its constraints. I found the techniques independently across
  the preceding work and claim no priority.

So this is **not a novel method**. It is (a) a small **engineering integration** — a pre-registered,
multi-gate pass/fail scorecard wired *inside* a running loop, with an adversarial critic step — and
(b) **negatives reported straight, then re-examined just as straight**: a later audit found one of the
"no edge" verdicts had been measured on a model fed corrupted input (a `== "v3"` version-gate bug
froze ~17% of its features), so that verdict was **withdrawn and is being re-measured on clean data**
(see `STORY.md` §4.5) — which does *not* imply an edge exists, only that the conclusion was not yet
earned. If there is any sliver of method not obviously covered above, it is treating *censoring
asymmetry* (an exit replay is right-censored at the realized close, so only tightening is observable)
as an explicit selection veto — and even that is offered tentatively, pending a direct read of
concurrent work.

The value here is not novelty. It is a worked, reproducible example of a self-improvement loop that
**refuses to fool itself in both directions** — rejecting false positives *and* withdrawing its own
comfortable negative once it turns out to rest on a bug — under exactly the conditions (noise,
non-stationarity, censoring) where documented systems have been shown to game their own evaluation.

## Files

| file | what |
|---|---|
| `evolve_lab.py` | true signal, data, mutation, both selection rules, evolution loop, `run_suite` |
| `selection_engine.py` | domain-agnostic gates + `select()` scorecard + `evolve()` loop |
| `domains.py` | domain adapters (symbolic regression; add one function per new domain) |
| `test_*.py` | 13 deterministic regression guards |
| `DESIGN.md` | architecture + roadmap |
| `STORY.md` | the narrative: the 1.5-year loop, the null result, and what survived |

## License

MIT (see `LICENSE`).

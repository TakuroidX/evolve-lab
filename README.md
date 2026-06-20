# evolve-lab

**A minimal, deterministic demonstration that the bottleneck in a self-improving loop is
*trustworthy selection*, not mutation — and that the same selection discipline that catches a
self-improving loop fooling itself can be made domain-agnostic and reused.**

Pure Python, zero dependencies, fully deterministic (fixed seeds). Independent of the trading
bot it was distilled from. For the longer narrative, see [`STORY.md`](STORY.md). 日本語は [`README.ja.md`](README.ja.md)。

```bash
python3 evolve_lab.py     # POC: contrast trustworthy vs naive selection across seeds
python3 -m pytest -q      # regression guards (evolve_lab 6 + selection_engine 11 + prompt_opt 11 + ab_select 9 + btc_exit 16 = 53 tests)
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
- **`gate_censoring`** (opt-in) — for domains with right-censoring, the improvement must hold on the *observed* (non-censored) subset; a gain that exists only on censored samples is rejected. Kept out of the defaults so it only bites domains that declare a `censored_key` (see the bot-exit section).

`select()` returns a **scorecard** (PASS / HOLD / FAIL) — it does not rank candidates (ranking invites
the winner's curse). `evolve()` runs the loop with a configurable acceptance gate. Adding a new domain
(`domains.py`) is one function; that is what "one engine" means.

```bash
python3 -c "import evolve_lab as el, selection_engine as se, domains; \
  print(el.mse(se.evolve(domains.symbolic_regression(), [0.0]*6, 300, 1, \
  accept_gates=[lambda d,c,i: se.gate_bootstrap(d,c,i,reps=400)])['incumbent'], \
  el.make_dataset(400,1.0,10000)))"   # engine generalizes on a gradient domain → test_mse ≈ 1.7
```

## A second domain: prompt optimization on a real LLM

`domains.prompt_opt` plugs a prompt-optimization task into the *same* engine — and unlike symbolic
regression it can run against a real LLM. The only file that touches the network is
`anthropic_backend.py` (stdlib `urllib`, key from env); tests use a deterministic fake, stay offline,
and remain CI-safe. A prompt is a system instruction: `variation` asks the LLM to rewrite it,
`evaluate` runs it on a labeled task and scores exact-match, `slice_key` is the task category.

Run against Claude Haiku (`demo_prompt_opt.py`, ~92 API calls, cents):

```
start   prompt : "answer the question"                                     score 0.000
evolved prompt : "answer the question with only the final value, no words" score 0.333  (1 accepted)
scorecard: FAIL   bootstrap pass · oos pass · regime FAIL (gain held in only one task category)
```

Two honest readings, both intended:

- The same engine **climbs on a real LLM** — Claude rewrote the prompt and the score rose from 0,
  with the improvement passing the bootstrap noise-floor and the time-block OOS gate.
- The engine then **refused to ship it**: `gate_regime` caught that the gain held in only one category
  (sign flip across slices) → overall **FAIL**. A real, calibrated *no* on a real model — the same
  "don't fool yourself" discipline, now exercised against an LLM rather than a synthetic gradient.

The score is modest **by design**: this demonstrates that the engine runs and that selection
discriminates on real-LLM output — it is *not* a tuned prompt optimizer. Prompt/agent optimization is
itself a crowded, adjacent field (DSPy, GEPA, ShinkaEvolve); the only thing claimed here is the
reusable *selection* discipline applied to it.

## A third domain: trustworthy A/B selection

`domains.ab_select` applies the *same* gates to A/B experiment analysis — no LLM, fully deterministic,
free to run. A *sample* is one unit `{week, segment, a, b}` (control vs treatment; the synthetic data
uses paired counterfactuals — real A/B is unpaired and would use the two-sample form of the *same*
gate logic). `naive_winner()` ships B whenever `mean(b) > mean(a)` — what teams routinely do. Four
planted scenarios, each a different way an A/B "win" can lie:

```
$ python3 demo_ab_select.py            # deterministic, no API, seed=1
scenario        naive           engine  boot  oos   reg   caught by
genuine         ship B (+5.14)   PASS   pass  pass  pass  — (true positive)
peeking         ship B (+2.54)   FAIL   pass  fail  pass  oos    (sign flips across time blocks)
concentrated    ship B (+2.15)   FAIL   pass  pass  fail  regime (win confined to one segment)
noise           ship B (+0.69)   FAIL   fail  fail  fail  bootstrap (within the noise floor)
```

The two **structural** false positives are robust, not cherry-picked: across 20 seeds, naive ships
`peeking` and `concentrated` **20/20** and the engine rejects them **20/20** — and crucially their
aggregate mean *passes* the bootstrap gate (the win looks real) while the structured gate catches it.
`noise` is honestly **probabilistic** — that is what noise *is*: naive ships it 16/20 seeds and the
bootstrap floor rejects it 19/20, so this is **not** claimed as "always 4/4." The honest, robust claim
is narrow and exact: **the two traps that fool aggregate statistics are caught every time, by the gate
that matches the trap** — the same engine and the same `select()` scorecard as the other two domains.

## The engine, turned back on the bot that produced it

`domains.btc_exit` is the loop closing: it makes the trading bot's **own exit logic** a domain of the
general engine. The bot's pure exit-replay functions (`replay_exit`, `load_paths`) are **faithfully
ported** (byte-identical bar docstrings) into evolve-lab, so it stays a standalone, dependency-free,
publishable repo — no coupling back to the bot. The public repo ships only a deterministic **synthetic
path fixture** (no trade history is published); locally the adapter reads the bot's real
`position_path_*.jsonl` read-only via `--paths`.

On **175 real SIM trades**, the general engine reproduces the bot's own exit-fitness verdicts — three
candidate exit changes, judged one at a time:

```
candidate              engine  boot  oos   reg   censored
早利確  tighter TS       PASS   pass  pass  pass   16.6%
早損切り tighter SL       FAIL   fail  fail  fail   12.6%
利伸ばし looser  TS       FAIL   fail  fail  fail   45.7%  → loosen + high-censored: vetoed
```

Three honest qualifications, stated up front:

1. **Verdicts agree; gates are not identical.** The selection here uses the *generic* three gates
   (bootstrap mean-CI / OOS / regime) plus a censoring surface — a re-expression of, not a copy of, the
   bot's `fitness.py` gates (median/outlier/payoff/censoring). The claim is that the verdicts **match**
   on the real run, not that the gate math is the same.
2. **PASS ≠ ship.** The `tighter TS` PASS is exactly the one the bot's *own* adversarial critic later
   rejected as premature (the gain was concentrated in a two-day trending window; the OOS split was
   noisy). The engine and the bot agree *because they run the same discipline* — including the part that
   refuses a PASS.
3. **Censoring does real work, not just decoration** — and it is a *generic gate*, not a hand-rolled
   veto. Price paths are right-censored at the realized close, so *loosening* an exit is structurally
   unobservable (you can't see past the close). `gate_censoring` doesn't crudely reject on a censored
   *rate* (that would wrongly kill a tightening that simply rarely binds); it asks whether the
   improvement **holds on the observed, non-censored subset**. On a constructed trap
   (`make_censoring_trap_paths`) a looser exit **passes bootstrap, OOS, and regime** yet has **zero
   observable samples** — so `gate_censoring` returns *insufficient* and downgrades the would-be PASS to
   **HOLD (don't ship)**, while a tightening candidate at 66 % censored still **passes** (its gain is
   observable). The asymmetry falls out naturally. This is the single most defensible idea in the repo,
   and it is *demonstrated*, not asserted.

```bash
python3 demo_btc_exit.py            # deterministic synthetic fixture (free) + the censoring-veto demo
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
| `domains.py` | domain adapters (symbolic regression + prompt optimization + A/B selection + bot exit; one function each) |
| `prompt_opt.py` | prompt-optimization domain: labeled task, scorer, deterministic fake model, cost-capped cache |
| `anthropic_backend.py` | the only networked file: real LLM model/rewrite via `urllib`, key from env, zero deps |
| `demo_prompt_opt.py` | runnable real-API demo (graceful no-cost skip when no key) |
| `ab_select.py` | A/B-selection domain: 4 planted scenarios + `naive_winner()` contrast (deterministic, no API) |
| `demo_ab_select.py` | deterministic, free contrast demo (naive mean-compare vs trustworthy scorecard) |
| `btc_exit.py` | bot exit-replay domain: faithfully-ported `replay_exit`/`load_paths`, synthetic fixture, censoring veto |
| `demo_btc_exit.py` | judges the bot's real exits read-only (`--paths`) or a synthetic fixture; censoring-veto demo |
| `test_*.py` | 53 deterministic regression guards (offline) |
| `DESIGN.md` | architecture + roadmap |
| `STORY.md` | the narrative: the 1.5-year loop, the null result, and what survived |

## License

MIT (see `LICENSE`).

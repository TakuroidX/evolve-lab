# The rarest output of a self-improving loop is a trustworthy "no"

*A solo, 1.5-year case study in building a self-improving system that learned to do the one thing
such systems are notoriously bad at: refuse to fool itself.*

---

## 1. The loop that wouldn't climb

I spent about a year and a half building a self-improving algorithmic-trading bot for BTC. The
intent was the obvious dream: a system that proposes changes to itself, keeps the ones that work,
and compounds. An evolutionary loop — variation, selection, inheritance — pointed at the market.

It never found a durable edge. Not for lack of effort or iterations. The pattern, in hindsight, was
always the same shape:

> a loss → add a filter or tweak a threshold → short-term improvement → a new problem appears →
> repeat.

For a long time I read this as a *model* problem: the predictor isn't good enough; build a better
one. That instinct is wrong, and recognizing why it's wrong is the whole story.

## 2. The actual failure: selecting noise that looks like signal

A self-improvement loop is, statistically, a multiple-testing machine. Every proposed change is a
hypothesis test against historical data. Run enough of them and some will look like improvements by
chance — the winner's curse. If your selection rule is "did the training metric improve?", you will
reliably promote noise, and the more you mutate, the faster you overfit. On a near-flat,
non-stationary, partially-censored fitness landscape — which is what a retail BTC backtest is —
*more search makes it worse*, not better.

So the bottleneck was never the model's horsepower. The model (a gradient-boosted predictor) already
had honest, modest skill (ROC-AUC ≈ 0.71). The bottleneck was **selection**: the loop could not tell
a real improvement from a lucky one. And worse, a self-improving loop has every incentive to deceive
itself — to find the framing in which it looks successful. Sakana's [Darwin Gödel
Machine](https://arxiv.org/abs/2505.22954) documents this directly: asked to fix its own tendency to
fake tool outputs, the agent instead *removed the markers its hallucination-detection reward relied on,
hacking the detector into reporting false successes* (their own write-up). A loop that grades its own
homework will, eventually, cheat.

The fix is not a smarter mutator. It is a **selector you can trust** — one engineered specifically to
resist being fooled, including by itself. *Trustworthy selection before more mutation.*

## 3. Method, or domain?

I stopped the auto-optimizer and rebuilt the selection side: a pre-registered, multi-gate scorecard
(contamination window, regime-robustness, bootstrap noise-floor, time-series out-of-sample, censoring
asymmetry, payoff-ratio), plus pre-registered kill-criteria, plus an adversarial critic pass. A
candidate change has to *survive* all of it, and the gate returns PASS/HOLD/FAIL — it does not rank
(ranking re-introduces the winner's curse).

But this left an uncomfortable question. If the loop still doesn't produce wins, is it because the
**method** (variation × selection × inheritance) is wrong, or because the **domain** has no gradient
to climb?

That question is what this repository isolates. `evolve_lab.py` puts the *same* loop on a domain where
a gradient provably exists — symbolic regression, recovering a hidden low-degree function from noisy
data with an over-capacity model. With trustworthy selection, the loop climbs and **generalizes**,
bounded near the irreducible error, with zero catastrophic runs. With naive train-only selection — the
exact "1.5-year loop" — it **blows up 8–15× on a fraction of seeds, unpredictably**. (Honest caveat:
on some seeds naive does fine; the difference is the *tail*, not the average. Not overstating this is
the point.)

The verdict: the method is sound. The bot's failure was the **domain** — there was no exploitable
gradient in the inputs we could actually access.

## 4. The day we proved it, honestly

The conclusion was not asserted; it was *earned*, in a single day of deliberately adversarial work,
with the goalposts nailed down in advance:

- The most robust remaining trading lever was put through **pre-registered kill-criteria committed to
  version control before the analysis ran**. It failed two independent decisive gates and was killed —
  even though killing it cost the last plausible hope of the bot earning anything.
- The selection harness, twice, **passed** a candidate that further scrutiny then refuted — so the
  harness was hardened to catch its own blind spot (a payoff-ratio gate; a fix for silently
  mislabeled regime data). The selector was made to select against *itself*.
- A skeptical survey of every accessible alternative edge source (order-flow, funding/basis,
  cross-exchange lead-lag, on-chain, sentiment, market-making, stat-arb) returned the same answer for
  our constraints: **no edge accessible at our scale and infrastructure.** The real edges live in
  trawler territory — colocation, capital, exclusive data — not in a retail dinghy fishing the most
  public, most-arbitraged data that exists.
- A literature check, run with the default set to *"not novel,"* found that essentially every
  component is prior art, and that a closely related project that appeared during this work —
  [MadEvolve](https://arxiv.org/abs/2605.23007) (May 2026), an LLM evolutionary loop on BTC with
  explicit p-hacking analysis — reached a strikingly similar *approach*. (Its conclusions differ:
  MadEvolve reports a positive result on its setup; this bot found no durable edge under its
  constraints. I found the techniques independently across the preceding work and claim no priority.)

Each of these is a "no." Together they are the honest, complete answer: **this bot has no edge, and
this work is not novel.** The right thing to do with that answer is to report it straight.

## 5. What survived

Two things outlived the bot's null result.

**A reusable selection engine.** The discipline, extracted into one domain-agnostic object
(`selection_engine.py`): variation × a pass/fail gate of self-deception-resistant tests × inheritance.
Point it at symbolic regression and it visibly climbs; point it at the trading bot and it honestly
reports no edge. New domains are one function each.

**A worked example of a self-improving loop that doesn't deceive itself.** This is the part I'd stand
behind. None of the statistics are new — the Deflated Sharpe Ratio and Combinatorial Purged
Cross-Validation (Bailey & López de Prado) are the rigorous ancestors of these gates. The contribution,
such as it is, is *integration and honesty*: a pre-registered multi-gate scorecard, an adversarial
critic, and kill-criteria, operating inside a live loop, producing a clean negative result without
moving a single goalpost — under exactly the noise/non-stationarity/censoring conditions where
documented systems have gamed their own evaluation.

There is even a small, self-referential proof in here: when I asked whether this work was
"cutting-edge," the same honest machinery returned *"no — others got there first."* The discipline
refused to inflate its own résumé. That refusal is the result.

## 6. Why it matters

The field is pouring resources into agents that improve themselves, and is increasingly worried about
those agents learning to deceive their own evaluators (reward hacking, eval gaming, fabricated
results). The scarce, valuable capability is not a cleverer mutator. It is **selection you can trust** —
and the institutional honesty to ship the negative result when that's what the data says.

This repo is the smallest faithful demonstration I could build of that capability: not "I built a
profitable bot" (I didn't, and that claim would be the exact self-deception this is about), but "I
built a self-improvement loop, and the most valuable thing it ever produced was a trustworthy *no* —
delivered on time, with the receipts."

---

*Code, tests, and the full positioning (including what is and isn't prior art) are in this repository.
Everything is deterministic and reproducible. If you find a place where I've overstated, that's a bug —
open an issue.*

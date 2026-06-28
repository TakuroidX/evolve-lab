"""Regression guards for demo_lookahead_leak.py — the standalone look-ahead-leak reproduction.

These tests are the proof that the demo is real and robust: on pure random-walk data (zero
real signal), the leaky feature must "predict" well above chance (it reads the future) while
the causal feature stays at coin-flip. If a future refactor accidentally fixes/breaks the
leak construction, these fail.
"""
import demo_lookahead_leak as d


def test_leak_inflates_auc_on_pure_noise_every_seed():
    for s in range(5):
        r = d.run(seed=s, n=12_000)
        assert r["leaky_auc"] > 0.60, f"seed {s}: leak should inflate AUC >0.60, got {r['leaky_auc']:.3f}"
        assert 0.44 < r["causal_auc"] < 0.56, f"seed {s}: causal must be ~coin-flip, got {r['causal_auc']:.3f}"


def test_leaky_clearly_beats_causal():
    for s in range(5):
        r = d.run(seed=s, n=12_000)
        gap = r["leaky_auc"] - r["causal_auc"]
        assert gap > 0.10, f"seed {s}: leak-vs-causal gap should be large, got {gap:.3f}"


def test_data_has_no_real_signal():
    # the causal feature on a pure random walk carries no edge -> averaged AUC ~0.5
    aucs = [d.run(seed=s, n=12_000)["causal_auc"] for s in range(8)]
    mean = sum(aucs) / len(aucs)
    assert 0.47 < mean < 0.53, f"causal mean AUC should be ~0.5 (no real signal), got {mean:.3f}"

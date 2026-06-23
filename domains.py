"""domains.py — selection_engine 用ドメインアダプタ。各ドメインを dict で返す。

エンジンは domain dict (variation/evaluate/data/ordered_key/slice_key) しか知らない。
新しいドメイン (BTC 出口・別最適化問題) はここに1つ関数を足すだけで載る = 「一つのエンジン」の証明。
"""
from __future__ import annotations

import ab_select as ab
import btc_exit as bx
import evolve_lab as el
import model_challenge as mc
import prompt_opt as po


def btc_exit(path_glob: str = None, n: int = 120, seed: int = 1, regime: str = "trend") -> dict:
    """BTC bot 出口リプレイ ドメイン (select 専用)。path_glob 指定=bot の実 logs を read-only 評価、
    未指定=決定論の合成 fixture。regime ∈ {trend, side, vol}。候補/現行は bx.CANDIDATES / bx.INCUMBENT。
    select(dom, bx.CANDIDATES["tight_sl"], bx.INCUMBENT) で 1候補ずつ判定。緩和候補は bx.censored_rate で veto。"""
    paths = bx.load_paths(path_glob) if path_glob else bx.make_synthetic_paths(n=n, seed=seed)
    regime_key = {"trend": bx.trend_phase, "side": bx.by_side, "vol": bx.by_vol}[regime]
    return bx.build_btc_exit_domain(paths, regime_key=regime_key)


def model_challenge(path_glob: str = None, samples: list = None) -> dict:
    """モデル訓練 (model-challenge) ドメイン (select 専用・依存ゼロ・xgboost 不要)。
    訓練済みモデルの pre-computed 予測を gate する: candidate (新版) vs incumbent (旧版)。
    path_glob 指定=bot 側が生成した eval_predictions*.jsonl を read-only ロード、samples 直渡しも可。
    select(dom, "candidate", "incumbent") で1回判定。delta=incumbent_logloss-candidate_logloss。
    見出しは model_challenge.headline_auc(dom["data"])。ordered_key=ts / slice_key=regime。"""
    if samples is None:
        if not path_glob:
            raise ValueError("model_challenge は path_glob か samples のどちらかが必須")
        samples = mc.load_predictions(path_glob)
    return mc.make_domain(samples)


def ab_select(scenario: str = "genuine", n: int = 120, seed: int = 1) -> dict:
    """A/B 選別ドメイン (select 専用・決定論・API 不要)。scenario ∈ ab_select.SCENARIOS。
    select(dom, "b", "a") で処理 B vs 対照 A を判定。ナイーブ比較は ab_select.naive_winner(dom)。"""
    return ab.build_ab_domain(ab.make_ab_samples(scenario, n=n, seed=seed))


def prompt_opt(model_fn=None, rewrite_fn=None, n: int = 24, seed: int = 1,
               max_calls: int = 100_000, max_rewrites: int = 100_000) -> dict:
    """プロンプト最適化ドメイン (本物の勾配 + ノイジー評価)。引数なし = 決定論 fake (API 不要)。
    実 API は anthropic_backend.make_model_fn()/make_rewrite_fn() を渡す (DESIGN.md Domain 2)。
    ⚠️ accept gate に gate_bootstrap を使うなら n>=10 必須 (未満は insufficient で受理ゼロ=登れない)。"""
    samples = po.make_prompt_samples(n=n, seed=seed)
    mf = model_fn or po.make_fake_model(samples)
    rf = rewrite_fn or po.make_fake_rewrite()
    return po.build_prompt_domain(mf, rf, samples, max_calls=max_calls, max_rewrites=max_rewrites)


def symbolic_regression(seed: int = 1, n_val: int = 60, noise: float = 1.0,
                        with_regime: bool = False) -> dict:
    """記号回帰ドメイン (勾配が本物=エンジンが実際に登れる対照)。
    sample = (x, y)。evaluate = 負の二乗誤差 (0 に近い=大きいほど良い)。"""
    val = el.make_dataset(n_val, noise, seed + 7)
    return {
        "variation": lambda coefs, rng: el.mutate(coefs, rng),
        "evaluate": lambda coefs, s: -(el.predict(coefs, s[0]) - s[1]) ** 2,
        "data": val,
        "ordered_key": None,  # 合成データは元の並びで OOS 分割
        # 合成 regime 軸: x<0 / x>=0。本物の改善なら両 slice で正のはず
        "slice_key": (lambda s: "NEG" if s[0] < 0 else "POS") if with_regime else None,
    }

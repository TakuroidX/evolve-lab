"""domains.py — selection_engine 用ドメインアダプタ。各ドメインを dict で返す。

エンジンは domain dict (variation/evaluate/data/ordered_key/slice_key) しか知らない。
新しいドメイン (BTC 出口・別最適化問題) はここに1つ関数を足すだけで載る = 「一つのエンジン」の証明。
"""
from __future__ import annotations

import evolve_lab as el


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

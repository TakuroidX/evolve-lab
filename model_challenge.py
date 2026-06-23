"""model_challenge.py — selection_engine 用「モデル訓練 (model-challenge)」ドメイン
(self-contained・依存ゼロ・決定論)。

狙い (model-challenge eval domain): 訓練済みモデルの「新版 (candidate) は旧版 (incumbent) より
本当に良いか」を、bot/A/B/出口と**同じ淘汰器**で裁く。これは dogfood = 既存の信頼できるゲートを
実 ML 問題 (確率予測の比較) にそのまま適用する。

🔴 依存ゼロが命: xgboost / numpy / pandas を **import しない**。モデルそのものは載せず、
   別タスク (bot 側) が holdout で生成した **pre-computed 予測** (incumbent_p / candidate_p) を
   read-only で gate する設計。evolve-lab は予測を「裁く側」であって訓練しない。

入力契約 (eval_predictions.jsonl・1行=1 holdout sample):
  {"ts": "2026-06-..T..", "regime": "UP|DOWN|NEUTRAL|...",
   "label": 0 or 1, "incumbent_p": 0.0-1.0, "candidate_p": 0.0-1.0}
  - label=1: その地平で価格 up / 0: down (flat は既に除外済)。
  - incumbent_p/candidate_p = 各モデルの p_up_cal 予測。
  - 既に out-of-sample (holdout)・時系列順。

ドメイン写像 (select 専用・A/B と同型):
  - evaluate(variant, sample) = -logloss(sample[variant + "_p"], label)   # 大=良い (負の logloss)
    → per-sample delta = eval("candidate") - eval("incumbent")
                       = incumbent_logloss - candidate_logloss            # 正=candidate が良い
  - select(dom, "candidate", "incumbent") で新版 vs 旧版を1回判定。
  - ordered_key = ts (時系列 OOS) / slice_key = regime (相場横断)。

logloss は予測を確率で評価する proper scoring rule (順位だけ見る AUC と違い較正も罰する)。
clip(p, 1e-6, 1-1e-6) で log(0) を防ぐ (安全な logloss)。

headline_auc は別途 rank-based AUC (xgboost 不要・純 Python) を提供 (見出し数字の sanity 用)。
gate は logloss delta を使う (確率/較正を見る) が、AUC は方向の識別力を補助的に surface する。
純Python・依存ゼロ・決定論 (seed 固定不要=確率/順位は決定論)。
"""
from __future__ import annotations

import glob
import json
import math

_EPS = 1e-6


# ---------------------------------------------------------------------------
# logloss (proper scoring・clip で安全に)
# ---------------------------------------------------------------------------
def _clip(p: float) -> float:
    return min(1.0 - _EPS, max(_EPS, float(p)))


def logloss(p: float, label: int) -> float:
    """1 sample の binary log loss (大きいほど悪い)。clip(p) で log(0) を防ぐ。"""
    p = _clip(p)
    return -(math.log(p) if label == 1 else math.log(1.0 - p))


def sample_delta(sample: dict) -> float:
    """per-sample 改善 = incumbent_logloss - candidate_logloss (正=candidate が良い)。"""
    return (logloss(sample["incumbent_p"], sample["label"])
            - logloss(sample["candidate_p"], sample["label"]))


# ---------------------------------------------------------------------------
# rank-based AUC (純 Python・xgboost/sklearn 不要・tie 対応)
# ---------------------------------------------------------------------------
def _auc(scores: list, labels: list) -> float:
    """Mann-Whitney U による ROC-AUC。tie は平均順位で扱う。正例/負例どちらか0件なら 0.5。"""
    pos = sum(1 for y in labels if y == 1)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    # 昇順ソートで平均順位を付与 (1-indexed)
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # 同値群の平均順位
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos_ranks = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    return (sum_pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)


def headline_auc(samples: list) -> dict:
    """incumbent_p / candidate_p それぞれの rank-based AUC (label 対比) + delta。
    見出し sanity 用 (gate は logloss delta を使う・AUC は識別力の補助 surface)。"""
    labels = [int(s["label"]) for s in samples]
    inc = _auc([s["incumbent_p"] for s in samples], labels)
    cand = _auc([s["candidate_p"] for s in samples], labels)
    return {"n": len(samples), "incumbent_auc": round(inc, 6),
            "candidate_auc": round(cand, 6), "delta_auc": round(cand - inc, 6)}


# ---------------------------------------------------------------------------
# 入力ロード (btc_exit.load_paths パターン・read-only・ts 昇順)
# ---------------------------------------------------------------------------
def load_predictions(path_glob: str) -> list:
    """eval_predictions*.jsonl (glob) を read-only でロードし ts 昇順の sample list に。
    必須キー (ts/label/incumbent_p/candidate_p) を欠く行と壊れ JSON はスキップ。regime は任意。"""
    out = []
    for fp in sorted(glob.glob(path_glob)):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not all(k in r for k in ("ts", "label", "incumbent_p", "candidate_p")):
                    continue
                out.append({
                    "ts": str(r["ts"]),
                    "regime": str(r.get("regime") or "UNKNOWN"),
                    "label": int(r["label"]),
                    "incumbent_p": float(r["incumbent_p"]),
                    "candidate_p": float(r["candidate_p"]),
                })
    out.sort(key=lambda s: s["ts"])
    return out


# ---------------------------------------------------------------------------
# ドメイン構築 (select 専用・variation は no-op = 比較であって進化でない)
# ---------------------------------------------------------------------------
def make_domain(samples: list) -> dict:
    """select() 専用 domain。candidate/incumbent は variant 名 "candidate"/"incumbent"。
    select(dom, "candidate", "incumbent") で新版 vs 旧版を1回判定。

    既存ゲートがそのまま通る形:
      - gate_bootstrap: evaluate の差 (= per-sample logloss delta) 列を bootstrap。
      - gate_oos: ordered_key=ts で時系列ブロック分割 (embargo 対応)。
      - gate_regime: slice_key=regime で相場スライス分割 (UNKNOWN は除外)。
    """
    samples = sorted(samples, key=lambda s: s["ts"])
    return {
        "variation": lambda variant, rng: variant,          # 未使用 (evolve しない)
        # 負の logloss = 大きいほど良い。delta = cand - inc = inc_logloss - cand_logloss
        "evaluate": lambda variant, s: -logloss(s[variant + "_p"], s["label"]),
        "data": samples,
        "ordered_key": lambda s: s["ts"],
        "slice_key": lambda s: s["regime"],
    }

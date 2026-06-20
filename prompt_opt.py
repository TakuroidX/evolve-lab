"""prompt_opt.py — selection_engine 用「プロンプト最適化」ドメインの中核 (タスク/スコアラ/キャッシュ/fake)。

狙い (DESIGN.md Domain 2): 市場と違って**本物の勾配がある**ドメインで、同じ淘汰エンジンが登ること、
そして「たまたま小 eval で勝っただけ/1 regime でしか効かない」変更を淘汰器が弾くことを実証する。

prompt は system 指示文字列。指示に含まれる「マーカー語」が、指示追従モデルの出力整形を改善し
ground truth と完全一致する確率を上げる = 勾配。マーカーは regime (タスク種別) ごとに効きが違う。

設計 (DESIGN.md「純コアを汚さない」):
  - model_fn(prompt, input) -> output / rewrite_fn(parent, rng) -> child を **注入式**に。
    実行 = anthropic_backend (別ファイル・urllib・追加依存ゼロ) / テスト = ここの決定論 fake。
  - 実 API コストを2系統とも封じる: 評価は CallCache (同一 (prompt,input) は1回計算) + max_calls 上限、
    変異(rewrite)は CallCounter で max_rewrites 上限 (rewrite はキャッシュ不可=毎回新プロンプト生成のため)。
純Python・依存ゼロ・決定論。
"""
from __future__ import annotations

import random

# 指示に現れうるマーカー語。useful 3 本は各 regime の整形を直す。useless 1 本はどの sample も要らない。
MARKERS = ["only", "concise", "number", "verbose"]
USELESS_MARKER = "verbose"

# regime (タスク種別) → その regime の sample が正解するために指示へ必要な単一マーカー
CAT_NEEDS = {"add": "concise", "extract": "number", "format": "only"}
CATEGORIES = list(CAT_NEEDS)

BASE_PROMPT = "answer the question"  # マーカーゼロ = 出発点 (どの sample も不正解)


def markers_in(prompt: str) -> set:
    p = prompt.lower()
    return {m for m in MARKERS if m in p}


def make_prompt_samples(n: int = 24, seed: int = 1) -> list:
    """ground truth 付き小タスクを生成。category を idx ラウンドロビンで均等配置
    (OOS の時間ブロック / regime スライスが偏らないように)。各 sample は単一マーカーを要する
    = マーカー1つ追加で一部 sample が flip する漸進的勾配になる (両方揃わないと動かない罠を避ける)。"""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        cat = CATEGORIES[i % len(CATEGORIES)]
        a = rng.randint(1, 99)
        if cat == "add":
            b = rng.randint(1, 99)
            inp, label = f"{a} plus {b}", str(a + b)
        elif cat == "extract":
            inp, label = f"ref {1000 + a} done", str(1000 + a)
        else:  # format
            inp, label = f"flag {a}", str(a)
        out.append({"idx": i, "input": inp, "label": label,
                    "category": cat, "needs": CAT_NEEDS[cat]})
    return out


def exact_match(output: str, label: str) -> float:
    """客観スコアラ: 完全一致なら 1.0 (大きいほど良い)。ノイズは小 eval + 整形揺れ由来。"""
    return 1.0 if str(output).strip() == str(label).strip() else 0.0


def make_fake_model(samples: list):
    """指示追従モデルの決定論シミュレータ: 指示に必要マーカーが在れば正しく整形 (=label)、
    無ければ整形崩れ (=不一致)。API 不要でエンジン/ゲートを検証するため。"""
    need_by_input = {s["input"]: s["needs"] for s in samples}

    def model_fn(prompt: str, input_text: str) -> str:
        label = next(s["label"] for s in samples if s["input"] == input_text)
        need = need_by_input[input_text]
        return label if need in markers_in(prompt) else f"{label} ?"  # ? = 整形崩れ→不一致
    return model_fn


def make_fake_rewrite():
    """決定論の変異: ランダムにマーカー1つを追加 (無ければ)、稀に除去。seed 付き rng で再現可能。"""
    def rewrite_fn(parent: str, rng: random.Random) -> str:
        m = rng.choice(MARKERS)
        if m in parent.lower():
            if rng.random() < 0.3:  # 稀に除去。空白は正規化 (二重空白でキャッシュキーが割れるのを防ぐ)
                return " ".join(parent.replace(m, "").split()) or "answer"
            return parent
        return f"{parent} {m}".strip()
    return rewrite_fn


class CallCache:
    """model_fn ラッパ: 同一 (prompt,input) は1回だけ計算 (実 API コスト削減) + max_calls 上限。
    上限超過は例外 (暴走防止)。calls = 実際に計算した unique 呼び出し数 (=コストの目安)。"""

    def __init__(self, fn, max_calls: int = 100_000):
        self._fn = fn
        self.max_calls = max_calls
        self.calls = 0
        self._store: dict = {}

    def __call__(self, prompt: str, input_text: str) -> str:
        key = (prompt, input_text)
        if key in self._store:
            return self._store[key]
        if self.calls >= self.max_calls:
            raise RuntimeError(f"CallCache: max_calls={self.max_calls} 超過 (コスト暴走防止)")
        v = self._fn(prompt, input_text)
        self._store[key] = v
        self.calls += 1
        return v


class CallCounter:
    """rewrite_fn ラッパ: 呼び出し数を数え max_rewrites で上限を掛ける (rewrite は毎回新出力なので
    キャッシュ不可・ここでしかコストを縛れない)。上限超過は例外 (コスト暴走防止)。"""

    def __init__(self, fn, max_rewrites: int = 100_000):
        self._fn = fn
        self.max_rewrites = max_rewrites
        self.calls = 0

    def __call__(self, parent, rng):
        if self.calls >= self.max_rewrites:
            raise RuntimeError(f"CallCounter: max_rewrites={self.max_rewrites} 超過 (コスト暴走防止)")
        v = self._fn(parent, rng)
        self.calls += 1
        return v


def build_prompt_domain(model_fn, rewrite_fn, samples: list, scorer=exact_match,
                        max_calls: int = 100_000, max_rewrites: int = 100_000) -> dict:
    """selection_engine が食う domain dict を組む。cost 計測/上限のため model キャッシュと
    rewrite カウンタを公開 ("_cache" / "_rewrites")。実 API の課金は両者の calls 合計が目安。"""
    cached = CallCache(model_fn, max_calls=max_calls)
    counted_rewrite = CallCounter(rewrite_fn, max_rewrites=max_rewrites)

    def evaluate(prompt: str, sample: dict) -> float:
        return scorer(cached(prompt, sample["input"]), sample["label"])

    def variation(prompt: str, rng: random.Random) -> str:
        return counted_rewrite(prompt, rng)

    return {
        "variation": variation,
        "evaluate": evaluate,
        "data": samples,
        "ordered_key": lambda s: s["idx"],     # OOS: idx 時系列ブロック
        "slice_key": lambda s: s["category"],  # regime: タスク種別横断
        "_cache": cached,                       # 評価 (model_fn) の unique 呼び出し数
        "_rewrites": counted_rewrite,           # 変異 (rewrite_fn) の呼び出し数
    }


def mean_score(domain: dict, prompt: str) -> float:
    ev = domain["evaluate"]
    data = domain["data"]
    return sum(ev(prompt, s) for s in data) / len(data)

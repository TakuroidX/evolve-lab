"""anthropic_backend.py — prompt_opt の「本物の LLM」アダプタ。**API を叩く唯一のファイル**。

DESIGN.md の原則: エンジン本体/テストは純Python・決定論を維持し、非決定論と外部依存はここに隔離する。
追加依存ゼロ (stdlib urllib のみ)。鍵は実行時に環境変数 ANTHROPIC_API_KEY から読む
(リポジトリに鍵を置かない・bot の .env に依存しない)。安価モデル既定 (claude-haiku-4-5)。

使い方 (実 API・コスト発生):
    import os, domains, selection_engine as se
    from anthropic_backend import make_model_fn, make_rewrite_fn
    dom = domains.prompt_opt(model_fn=make_model_fn(), rewrite_fn=make_rewrite_fn(), n=8, max_calls=400)
    ...
テスト/CI は domains.prompt_opt() を引数なしで呼べば決定論 fake が入る (このファイルは import されない)。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # 安価モデル (コストガード)
RETRY_STATUS = {429, 500, 502, 503, 529}     # 一過性エラーは小回数だけ再試行

# 変異の多様性のための seed 付き directive (rewrite が1点だけ変える指示)
REWRITE_DIRECTIVES = [
    "make the output formatting instruction stricter",
    "tell it to answer with only the final value, no words",
    "remove any chance of extra explanation",
    "clarify the expected answer format for numbers",
    "make it more concise",
]


def _post(api_key: str, model: str, system: str, user: str, max_tokens: int,
          retries: int = 3) -> str:
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    })
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
            return "".join(parts).strip()
        except urllib.error.HTTPError as e:
            if e.code in RETRY_STATUS and attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def _require_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY が環境変数に未設定。実 API デモは鍵を export してから実行してください "
            "(リポジトリには鍵を置かない)。テストは決定論 fake で API 不要。")
    return key


def make_model_fn(model: str = DEFAULT_MODEL, max_tokens: int = 32):
    """model_fn(prompt, input) -> output: prompt を system 指示として input を解かせる。"""
    key = _require_key()

    def model_fn(prompt: str, input_text: str) -> str:
        return _post(key, model, system=prompt, user=input_text, max_tokens=max_tokens)
    return model_fn


def make_rewrite_fn(model: str = DEFAULT_MODEL, max_tokens: int = 256):
    """rewrite_fn(parent, rng) -> child: 既存プロンプトを1点だけ改良した版を生成 (seed 付き directive)。"""
    key = _require_key()

    def rewrite_fn(parent: str, rng) -> str:
        directive = REWRITE_DIRECTIVES[rng.randrange(len(REWRITE_DIRECTIVES))]
        meta = (f"You improve task system-prompts. Here is one:\n---\n{parent}\n---\n"
                f"Rewrite it changing exactly one thing: {directive}. "
                f"Output ONLY the rewritten prompt, no preamble.")
        child = _post(key, model, system="You are a precise prompt editor.",
                      user=meta, max_tokens=max_tokens)
        return child or parent
    return rewrite_fn

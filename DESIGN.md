# selection-engine — 設計 (2026-06-19, interview-build)

「信頼できる淘汰エンジン」を **一つのオブジェクト**として育てる (user 2026-06-19「このエンジンを
一つのオブジェクトとして進化しようぜ」)。BTC bot の exit fitness harness と evolve-lab の進化ループ
から核を抽出し、ドメイン非依存にした。

## インタビュー確定 (interview-build, 全て推奨を選択)

| 分岐 | 決定 |
|------|------|
| エンジンの正体 | **ドメイン非依存の淘汰エンジン**: 変異×淘汰(汎用ゲート)×遺伝。fitness とデータを plug すれば BTC でも記号回帰でも次の何でも回る |
| 棲む場所 | **evolve-lab と統合した独立リポ** (bot から切り離した公開可能な札。bot はこのエンジンの「一例」になる) |
| 第一目的 | **次のシステムで再利用する道具** 優先 (動く・テスト堅い・自分が使う)。公開品質は後で足す |

## アーキテクチャ

```
domain (dict) ── variation(parent,rng)->candidate
              ── evaluate(candidate,sample)->float   # 大きいほど良い
              ── data: [sample]                       # held-out 評価標本
              ── ordered_key(sample)  (任意, OOS 用)
              ── slice_key(sample)    (任意, regime 用)

selection_engine.py
  汎用ゲート: gate_bootstrap / gate_oos / gate_regime   # 自分を騙さない3本
  select(domain, cand, inc, gates) -> Verdict(PASS/HOLD/FAIL, [GateResult...])  # ranking しない
  evolve(domain, incumbent, generations, accept_gates) -> {incumbent, accepted} # 変異×淘汰×遺伝

domains.py
  symbolic_regression(...) -> domain   # 勾配が本物=エンジンが実際に登れる対照 (実証済 test_mse≈1.7)
  (今後) btc_exit(...) -> domain        # bot の price path を載せる adapter
```

## 設計原則 (bot 1.5年 + 2026-06-19 の結論から)
- **変異の前に信頼できる淘汰を**。進化の質は fitness の質で決まる。平坦/ノイズ地形で変異=過学習ドリフト。
- ゲートは **scorecard** であって **ranking でない** (bot Lesson-5)。**PASS≠ship** (Lesson-6: 最終は人/critic)。
- ループは安く (accept_gates=軽い)、出荷判断は厳格 (select フルスコアカード + 人/critic) = bot 運用と同型。
- "UNKNOWN" スライスは本物の regime として数えない (2026-06-19 /critic で発見した盲点を恒久化)。

## 実証済 (test_selection_engine.py 7件 PASS)
記号回帰ドメインで [0]*6 から evolve → test_mse 1.68 (既約≈1.0) = **同じエンジンが勾配ドメインで汎化**。

## ロードマップ (再利用する道具 として)
1. ✅ 汎用エンジン core + 汎用3ゲート + symbolic domain + tests (2026-06-19)
2. ✅ README 英語化 + 公開品質 (2026-06-20, 公開判断は user のみ)
3. ✅ **prompt_opt ドメイン adapter** (2026-06-20, 実 API 実走済) ← 第2の「本物の勾配」ドメイン
4. ✅ **ab_select ドメイン adapter** (2026-06-21, A/B 選別の規律: 決定論・select 専用・敵対レビュー済)
5. ✅ **btc_exit ドメイン adapter** (2026-06-21, replay_exit/load_paths 忠実移植 + 合成fixture +
   censoring veto 実演。実 175-path で bot fitness 判定と一致・敵対レビュー済) ← 生まれ故郷へエンジンを還す
6. ✅ **censoring を汎用ゲート化** (2026-06-21, `gate_censoring` を selection_engine 本体へ。打ち切り「率」でなく
   **観測可能部で改善が残るか**で判定=方向非対称が自然に出る・tighten 誤殺なし。trap で PASS→HOLD 降格を実証)
   / ☐ payoff比ゲートの汎用化 (まだ btc/bot 固有)

## Domain 2: prompt_opt 設計 (2026-06-20, interview-build)

**狙い**: 「同じ淘汰エンジンが、市場と違って**本物の勾配があるドメイン**でも登る」を実証する第2例。
プロンプト最適化は (a) 勾配が実在 (b) 評価がノイジー (小eval/サンプリング揺れ) = ナイーブ選別なら
「たまたま小evalで勝ったプロンプト」を出荷してしまう。淘汰器がそれを弾く様子を見せる。

| 分岐 (interview 2026-06-20) | 決定 |
|------|------|
| 評価/変異の回し方 | **Anthropic API 主体** (最高忠実度・Sakana 直結。コスト/非決定論は下記ガードで封じる) |
| 着手順 | **2 (prompt_opt) 先行 → 3 (ab_select)** |

**設計原則 (純コアを汚さない)**:
- エンジン本体 (`selection_engine.py`) と全テストは **純Python・依存ゼロ・決定論** を維持。
- API を叩くのは **`anthropic_backend.py` 1ファイルだけに隔離** (stdlib `urllib` のみ・**追加依存ゼロ**)。
  鍵は実行時に `ANTHROPIC_API_KEY` 環境変数から読む (リポジトリに鍵を置かない・bot の .env に依存しない)。
- `model_fn(prompt, input) -> output` / `rewrite_fn(parent, rng) -> child` を **注入式**に。
  実行=Anthropic backend / **テスト=決定論の fake** (API 不要のまま CI 可能・再現性確保)。
- **コストガード**: 評価キャッシュ (同一 (prompt,input) は1回だけ) + max_calls 上限 + 安いモデル
  (claude-haiku-4-5)。実 API デモは **user の明示 OK 後のみ** 実行 (推定コストを先に提示)。

**ドメイン dict の対応**:
- `variation(parent_prompt, rng)` = LLM がプロンプトを1点改変 (seed 付き directive)。
- `evaluate(prompt, sample)` = prompt で sample.input を解かせ、**ground truth** と突合したスコア
  (客観スコアラ。ノイズは小eval + サンプリング揺れ由来 = 自己欺瞞リスクが出る所)。
- `data` = リポジトリ同梱の小 eval セット (input/label/category)。
- `slice_key` = category (regime ゲート: 1カテゴリだけで勝つ過学習を弾く)。`ordered_key` = index。

**正直な scope (誇張回避)**: これは「同じエンジンが勾配ドメインで汎化する」POC であって製品ではない。
PASS≠ship (Lesson-6)。スコア改善≠実用 (本番タスクは別)。bot の no-edge とは独立 (こちらは勾配が在る)。

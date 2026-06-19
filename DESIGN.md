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
2. ☐ btc_exit ドメイン adapter (bot の position_path を load → 既存 fitness.py を本エンジンに載せ替え検証)
3. ☐ payoff比 / censoring ゲートを汎用化して plug (bot 固有から汎用へ)
4. ☐ README 英語化 + 公開品質 (Sakana 札・公開判断は user のみ)

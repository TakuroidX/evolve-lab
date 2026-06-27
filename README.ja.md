# evolve-lab (日本語)

**「変異 × 信頼できる淘汰 × 遺伝」を、勾配が本物のドメインで動かす。**

純 Python・依存ゼロ・決定論 (seed 固定)。BTC bot リポジトリとは無関係の独立プロジェクト。
英語版・物語は `README.md` / `STORY.md`。

> **これは何で、何でないか。** これは **case study** と **正直な null result** であって、novel な手法でも
> SOTA ツールでも、ましてや儲かる bot でもない。個々の技術はすべて既存のより厳密な形がある(Deflated
> Sharpe Ratio / PBO / CPCV — 下記「正直なポジショニング」)。貢献は *統合*(自己改善ループの中に pass/fail
> scorecard を配線したこと)と *正直さ*(negative をそのまま出すこと)。私有データ不要で一番分かりやすい
> デモは **`python3 demo_ab_select.py`** — A/B の「勝ち」が嘘をつく3通りを、合成データ上で数秒でゲートが
> 捕捉する様を見せる。
>
> **再現性の範囲**: 公開コードと合成 fixture は完全に決定論的で再現可能。bot の実トレード履歴とモデル
> 訓練/live 評価は private upstream にあり、それらの具体値(N=175 出口・model-rebuild AUC 等)は
> **報告のみで、ここでは独立に再現できない**。

2部構成:
- **`evolve_lab.py`** — 最小 POC: 3つの選別ルール(naive=train / plain=held-out / gated=held-out+ゲート)を対比し、**裾の破滅を断つのは held-out 評価そのもの**(ゲートでなく)であることを正直に示す (下記)。
- **`selection_engine.py`** — そこから抽出した **ドメイン非依存の「信頼できる淘汰エンジン」** (一つのオブジェクト)。
  variation×淘汰(汎用ゲート: bootstrap/OOS/regime/censoring)×遺伝 を、fitness とデータを plug すればどのドメインでも回せる形に。`domains.py` に **5 ドメイン**(記号回帰 / prompt 最適化 / A/B 選別 / bot 出口 / model 再構築判定)が各1関数で載る。設計: `DESIGN.md`。
  ⚠️ これらのゲートは Deflated Sharpe Ratio / PBO / CPCV (López de Prado) の **依存ゼロ軽量再実装**で、統計的厳密性は劣る(多重検定補正なし・purge は端切りのみ)。本気の検定なら mlfinpy / LdP を使うこと。価値は手法の novelty でなく統合と正直な negative。

```bash
python3 evolve_lab.py                       # POC デモ (seed 1-6 を集計)
python3 demo_ab_select.py                    # 私有データ不要・中心デモ (ゲートが偽陽性3型を捕捉)
python3 -m pytest -q                         # 回帰ガード (evolve_lab 7 + selection_engine 12 + prompt_opt 11 + ab_select 9 + btc_exit 16 + model_challenge 9 = 64 件)
```

## なぜ作ったか
BTC trading bot を 1 年半育てる中で、自動最適化ループ (proposer) を**意図的に停止**した。
理由は「市場には信頼できる勾配 (エッジ) がほぼ無く、平坦な地形で変異率を上げると過学習ドリフト
にしかならない」から。── *変異の前に、信頼できる淘汰を*。

疑問: **方法 (変異×淘汰×遺伝) が悪いのか、ドメイン (市場) が悪いのか?**
この POC は後者だと示す。**勾配が在るドメインなら、同じループは実際に登る(held-out 選別で)。**
そして「train だけ見るナイーブ淘汰」(= 1y3m ループの正体) は、**裾で破滅的に過学習する**。

## 結果 (seed 1-6, 決定論的に再現)
```
既約 test_mse ≈ 1.0  (ノイズ分散。これ未満は原理上不可能)
  seed   naive(train)  plain(held-out)  gated(held-out)
   1       8.98          3.44             1.35   ← ナイーブ破滅
   2       1.49          1.25             1.92
   3      15.46          1.53             1.59   ← ナイーブ破滅
   4       2.84          1.54             3.95
   5       1.93          1.23             2.75
   6       6.59          1.34             2.56   ← ナイーブ破滅
naive (train のみ): 中央 4.72 / 最悪 15.46 / 破滅 3 件  → train だけ見る罠
plain (held-out)  : 中央 1.43 / 最悪  3.44 / 破滅 0 件  → held-out を使うだけで裾を断つ
gated (held-out)  : 中央 2.24 / 最悪  3.95 / 破滅 0 件  → この toy では plain を上回らない
```
> 正直な注記(+ 自己訂正): 旧版は naive(train) vs gated(held-out) だけを比べ「**ゲート**が裾を断つ」としていたが、
> これは**交絡**(データと選別ルールを同時に変えていた)。データを揃える(plain 列)と、**裾を断つのは
> held-out データそのもの**でゲートではないと分かる。この単純 toy の失敗モードは小標本過学習だけで、
> plain held-out で既に捕まる。ゲートが plain を上回るのは「多重比較 / regime 偏り / 打ち切り」設定で、
> それは `ab_select` / `btc_exit` が交絡なしに実証する。看板デモの交絡を公開前に自分で捕まえたこと自体が
> thesis の実演(`STORY.md` §4.5 と同じ規律)。seed 2/5 はナイーブも汎化する。誇張しない。

## bot との対応 / 結論
ナイーブ淘汰の破滅 = bot の「損失→フィルター追加→短期改善→別の問題」ループ。
信頼淘汰 = bot の `fitness.py` (contamination/regime/bootstrap/censoring/OOS) の一般化。
**bot が進化しなかったのは "方法" でなく "市場に勾配が無い" せい。** 勾配が在れば同じ規律で登る、騙されずに。

## 正直なポジショニング (何で、何でないか)
個々の技術はどれも novel でない:
- **backtest 過学習統計**: Deflated Sharpe Ratio / PBO / Combinatorial Purged CV (Bailey & López de Prado) が、ここの bootstrap/OOS/contamination ゲートの**厳密で上位の形**。
- **自己改善エージェント**: Darwin Gödel Machine (Sakana/UBC)・AlphaEvolve (DeepMind)。自己改善ループ=多重検定で自分を騙しうる、という枠組み。
- **BTC×LLM進化 specifically**: [MadEvolve](https://arxiv.org/abs/2605.23007) (Kvasiuk et al., 2026-05) が時系列 train/val/test + 明示的 p-hacking 解析で *approach 的に酷似*。結論は逆(MadEvolve は positive・本 bot は no-edge)。技術は先行研究から独立に到達したもので priority は主張しない。

= **novel な手法ではない**。価値は (a) 小さな**工学的統合**(進化ループ内に配線した pass/fail scorecard + 事前登録 kill + 敵対 critic)と (b) **正直な negative**(偽陽性を却下し、かつ自分の心地よい結論すらバグ発覚で撤回し、データで再 earn した二方向の正直さ)。唯一 narrow に残る角度は censoring 非対称 veto(出口リプレイは実現 close で右打ち切り=tightening だけ観測可能)を選別 veto に使う点で、それも暫定的に提示。本物の差別化は手法でなく **judgment と integrity**(キャリア文脈の札)。誇張すれば唯一の差別化=正直さが死ぬ。

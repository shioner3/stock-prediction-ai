# PHASE 8 REPORT: long_oversold_rebound 再現性・BEAR regime依存性の追加検証

実行日: 2026-08-20
config_hash: `a9b34ccb6e6e1e1a9d4d1eb554fe3fd8a4c09b2d9d186b2bd0ee3b0d430a2217`
(CONFIG_MISMATCH検証: Phase 6.5・Phase 7・現在の3値が完全一致することを
`pipeline/run_phase8_analysis.py::verify_config_hash()`で機械的に確認済み)

データ範囲: 2022-01-04 〜 2026-08-20(Phase 7が既に取得済みの連続データセット
`data/phase7/`をそのまま再利用。新規Fetchなし)
Universe: 2,880銘柄(Phase 7と同一)

---

## 0. Executive Summary

**結論を一言で言うと: 「方向性としては再現するが、統計的証拠のかなりの部分が
2024年8月の1回の市場イベント(9営業日)に集中しており、額面通りの数値を
そのまま『安定した優位性』と解釈するのは危険」。**

| 項目 | 値 |
|---|---|
| Phase 6.5結果 | PF(base)=1.160, Decision=INSUFFICIENT_EVIDENCE |
| Phase 7結果 | PF(base)=1.800, Decision=ACCEPT_CANDIDATE |
| **Combined OOS結果**(2022-01〜2026-08通し、全14 Window) | PF(base)=1.585, Expectancy=+0.0107, n=8,559, **Decision=ACCEPT_CANDIDATE** |
| BEAR regime限定PF(base) | **34.9**(全regime中で圧倒的に高い) |
| **最重要所見** | BEAR regimeの累積リターンの**71.6%**が、たった1つのBEAR
  episode(2024-08-02〜08-15、9営業日、2024年8月の実際の市場急落・急反発
  イベント)から来ている |
| Leave-One-Period-Out | この1 episodeを除いてもPFは1を割らない(39.6→14.9)
  が、**62%の低下**は無視できない |
| Placebo(同一Signal・同一銘柄を15営業日前にずらして再実行) | BEAR regime
  PF=0.095(実際のSignalの1/400以下) - タイミングそのものに意味がある
  ことを支持 |
| 銘柄集中度 | Top20銘柄でtrade share 5.3%、return share 7.7% - 銘柄集中は
  **確認されない** |
| 最終分類 | **R1条件を字面上は満たすが、単一イベント集中という重大な
  留保付き**(下記§10で詳述) |

**実運用への自動採用は行っていない。** 本Phaseの目的も達成: 「BEAR regime
という条件付きの優位性が本当に再現性のある現象か」を検証した結果、
「ある程度は再現するが、その大部分は2024年8月の特定の歴史的イベントに
起因しており、"BEAR regimeなら常に効く"という単純な解釈は支持されない」
という、より precise な結論が得られた。

---

## 1. Q1〜Q8 への回答

### Q1. Phase 6.5でもPhase 7でも再現したか?
**部分的に再現。** 両フェーズともPF(base)>1(1.160→1.800)で方向性は一致した
が、Phase 6.5単独ではDecision Frameworkの全基準(特にHigh cost tierでの
expectancy>0)を満たせずINSUFFICIENT_EVIDENCEだった。Phase 7・Combinedでは
全基準を満たしACCEPT_CANDIDATEとなった。

### Q2. 優位性はBEAR regimeに限定されるか?
**Yes、明確に限定される。** Combined regime別PF: BEAR=42.8(OOS window内
限定)、BULL=1.01、NEUTRAL=1.08。Phase 6.5単体・Phase 7単体でもBULL/NEUTRAL
は概ね1.0前後で推移しており、BEAR以外での優位性はほぼ存在しない。

### Q3. BEAR regime内でも複数期間・複数Windowで再現するか?
**方向性としては再現するが、規模は極端に不均一。** 取引のあった11の
BEAR episode(トレードあり)中9episodeでPF>1、5つのWFO Windowで取引が発生し
全てでも大半がPF>1(Window 6のみPF=0.31で唯一の例外)。ただし累積リターンの
71.6%が単一episode(2024-08-02〜08-15)に集中しており、「均等に複数期間で
再現している」とは言い難い。詳細は§5参照。

### Q4. 特定銘柄への依存はないか?
**依存なし。** BEAR regime限定でUnique銘柄数1,014、Top20銘柄のtrade share
はわずか5.3%、return shareも7.7%。少数銘柄が結果を作っているという懸念は
支持されない(§7)。

### Q5. High Cost TierでもPF>1を維持するか?
**維持する、大幅に。** BEAR限定High cost(80bps)でもPF=28.07(zero costの
39.57から大きく下がるが、依然として1を大幅に上回る)。Combined全体でも
High cost PF=1.27で1を超える(§6)。

### Q6. Bootstrap CIでExpectancy > 0が維持されるか?
**維持する。** BEAR限定: Expectancy 95% CI = [0.0852, 0.0943](ゼロを大きく
上回って安定)。Combined全体: CI = [0.00918, 0.01227]。いずれもゼロを跨がない。

### Q7. Permutation Test + FDR補正後でも有意性が維持されるか?
**表面上は維持するが、重要な統計的留保がある。** BEAR限定permutation
p=0.0000(10,000回中1回も観測値を超えなかった)。Combined全体でも
p=0.0258。Phase 7単独でのFDR補正後p値は0.0000(rank 1/12、有意)。

**ただし**: BEAR regime限定のpermutation testは、n=1,226のサンプルの
53.2%が同一9営業日イベント由来であり、これらの観測値は独立試行ではなく
強く相関している(同じ市場イベントに対する多数銘柄の反応)。Permutation
Testが仮定する「観測が交換可能である」という前提が実質的に破られており、
p値が真の有意水準を過小評価(=過度に楽観的)している可能性が高い。
**額面通りのp=0.0000を鵜呑みにすべきではない。**

Combined全体の新規12-Signal FDR補正は本レポート作成時点でJSON保存対象に
含めておらず(§11「軽微な記録漏れ」参照)、正確な値は未取得。Phase 7単独の
FDR結果(有意)とPhase 6.5単独の結果(不十分)から、Combinedはその中間
またはPhase 7寄りの結果になると推測されるが、正確な値は将来の再実行で
補完すべき。

### Q8. 特定のBEAR episodeを除外すると優位性が消えるか?
**「消える」とまでは言えないが、大きく減少する。** Leave-One-Period-Out
分析(§9)で、2024-08-02〜08-15のepisodeを除外すると、BEAR regime全体の
PFは39.57→14.90に低下(62%減)。本レポートのLOPO分類ルール(PF>1→PF<1に
反転した場合のみPERIOD_DEPENDENT)では「STABLE」と分類されるが、これは
判定ルールが荒すぎることによる過小評価であり、**実質的には強い期間依存性が
存在する**と解釈すべきである(詳細な数値は§9で全て開示)。

---

## 2. Combined OOS(Phase 6.5 + Phase 7 統合、全14 Window)

| Cost Tier | Trades | PF | Expectancy | Win Rate |
|---|---:|---:|---:|---:|
| Zero | 8,559 | 1.809 | +0.01371 | 55.5% |
| Low | 8,559 | 1.731 | +0.01271 | 55.2% |
| Base | 8,559 | 1.585 | +0.01071 | 53.1% |
| High | 8,559 | 1.275 | +0.00571 | 47.8% |

- Bootstrap Expectancy 95% CI: [0.00918, 0.01227](ゼロを跨がない)
- Permutation p = 0.0258(n_signal=14,511、n_population=2,281,980)
- Windows with PF>1: 9/14
- **Decision: ACCEPT_CANDIDATE**(既存Decision Frameworkによる機械的判定、
  変更なし)

Combined OOSはPhase 6.5(2,755銘柄)・Phase 7(2,880銘柄)それぞれの保存済み
JSONを単純に足し合わせたものではなく、**Phase 7が保有する連続データセット
(2,880銘柄、2022-01-04〜2026-08-20)上で`run_walk_forward(min_oos_start=None)`
を再実行**した結果である。これにより、異なるUniverseサイズの2つの結果を
混在させることなく、単一の一貫したUniverseで14 Windowを評価している。

---

## 3. WFO Window一覧(Combined、全14)

Window 0〜4はPhase 6.5のOOS期間、Window 5〜13はPhase 7のOOS期間と日付が
完全一致(既存WFO生成ロジック`backtest/walk_forward.py::generate_windows()`
を無変更で再利用、`min_oos_start`パラメータで期間別に絞り込み可能 -
`pipeline/run_walk_forward.py`に追加した唯一の新規パラメータ)。

| Window | OOS Start | OOS End |
|---|---|---|
| 0 | 2023-04-04 | 2023-07-04 |
| 1 | 2023-07-04 | 2023-10-04 |
| 2 | 2023-10-04 | 2024-01-04 |
| 3 | 2024-01-04 | 2024-04-04 |
| 4 | 2024-04-04 | 2024-07-04 |
| 5 | 2024-07-04 | 2024-10-04 |
| 6 | 2024-10-04 | 2025-01-04 |
| 7 | 2025-01-04 | 2025-04-04 |
| 8 | 2025-04-04 | 2025-07-04 |
| 9 | 2025-07-04 | 2025-10-04 |
| 10 | 2025-10-04 | 2026-01-04 |
| 11 | 2026-01-04 | 2026-04-04 |
| 12 | 2026-04-04 | 2026-07-04 |
| 13 | 2026-07-04 | 2026-08-20(短縮) |

---

## 4. BEAR × WFO Window

| Window | OOS期間 | BEAR Trades | PF |
|---|---|---:|---:|
| 0〜4 | 2023-04〜2024-07 | 0 | - |
| 5 | 2024-07〜2024-10 | 718 | 54.10 |
| 6 | 2024-10〜2025-01 | 22 | 0.31 |
| 7 | 2025-01〜2025-04 | 9 | 2.48 |
| 8 | 2025-04〜2025-07 | 393 | 95.00 |
| 9〜10 | 2025-07〜2026-01 | 0 | - |
| 11 | 2026-01〜2026-04 | 14 | 3.01 |
| 12〜13 | 2026-04〜2026-08 | 0 | - |

Window 0〜4(Phase 6.5期間)にBEAR取引が1件も無いのは、Phase 6.5の期間中
BEAR判定日そのものが0件だったため(既報告済みの制約)。Window 6のみPF<1
(0.31)で、唯一の「BEARなのに負けている」Window。

---

## 5. BEAR × 年別 / BEAR Episode(重要セクション)

### 5.1 年別

| 年 | BEAR Trades | PF |
|---|---:|---:|
| 2022 | 76 | 10.78 |
| 2023 | 0 | NO_BEAR_DATA |
| 2024 | 737 | 38.58 |
| 2025 | 399 | 78.82 |
| 2026 | 14 | 3.01 |

### 5.2 BEAR Episode(連続BEAR判定日の区間、全期間で13件)

| # | 期間 | 営業日数 | Trades | PF | 累積リターン |
|---|---|---:|---:|---:|---:|
| 0 | 2022-04-11〜04-12 | 2 | 11 | 2.09 | +0.059 |
| 1 | 2022-06-21〜06-23 | 3 | 65 | 12.68 | +2.862 |
| 2 | 2022-07-01 | 1 | 0 | - | - |
| **3** | **2024-08-02〜08-15** | **9** | **652** | **130.96** | **+81.419** |
| 4 | 2024-08-19 | 1 | 1 | ∞ | +0.003 |
| 5 | 2024-08-21〜08-22 | 2 | 2 | 0.09 | -0.033 |
| 6 | 2024-08-27〜08-29 | 3 | 0 | - | - |
| 7 | 2024-09-04〜09-26 | 15 | 53 | 2.17 | +0.824 |
| 8 | 2024-09-30〜10-04 | 5 | 10 | 1.78 | +0.143 |
| 9 | 2024-10-08〜10-16 | 6 | 19 | 0.23 | -0.481 |
| 10 | 2025-04-03〜04-17 | 11 | 398 | 78.59 | +28.597 |
| 11 | 2025-04-21〜04-24 | 4 | 1 | ∞ | +0.085 |
| 12 | 2026-03-30〜03-31 | 2 | 14 | 3.01 | +0.204 |

**Episode 3(2024-08-02〜08-15、9営業日)は、2024年8月に実際に発生した
市場急落・急反発イベント(日経平均の史上最大級の1日下落とその翌日以降の
急反発)に対応する。個別銘柄の生OHLCVを直接確認し、株式分割等のデータ
異常が原因でないことを検証済み(§11参照、コード上のバグではない)。**

このepisode単独で:
- **全BEAR取引数の53.2%(652/1,226)**
- **全BEAR累積リターンの71.6%(81.4/113.7)**

を占める。Episode 10(2025-04-03〜04-17、11営業日、398取引、+28.6累積
リターン)も同様に大きいが、Episode 3ほどではない。この2つのepisodeだけで
BEAR累積リターンの約96%(81.4+28.6=110.0 / 113.7)を占めており、
「BEAR regimeという条件全般で安定して効く」というより「特定の急落・
反発イベントで極めて大きく効く」という性質に近い。

---

## 6. Cost Sensitivity(BEAR regime限定)

| Tier | PF | Expectancy |
|---|---:|---:|
| Zero | 39.57 | +0.0927 |
| Low | 37.96 | +0.0917 |
| Base | 34.89 | +0.0897 |
| High | 28.07 | +0.0847 |

コストを最大(80bps)にしてもBEAR限定PFは28超と極めて高い。ただし、これは
Episode 3の極端な値に強く引きずられた数値である点に注意(§5参照)。

---

## 7. Concentration(BEAR regime限定)

| Top-K | Trade Share | Return Share |
|---|---:|---:|
| Top1 | 0.41% | 0.64% |
| Top5 | 1.71% | 2.54% |
| Top10 | 2.94% | 4.38% |
| Top20 | 5.30% | 7.74% |

Unique銘柄数1,014。**銘柄集中は確認されない** - 上記の極端な数値は特定
銘柄ではなく特定の**時間帯(Episode 3等)**に起因することが§5との対比で
明確になる。

---

## 8. Bootstrap / Permutation(BEAR regime限定)

| 指標 | Point Estimate | 95% CI |
|---|---:|---|
| Mean Return | 0.0897 | [0.0852, 0.0943] |
| Profit Factor | 34.89 | [26.76, 47.07] |
| Expectancy | 0.0897 | [0.0852, 0.0943] |

Permutation p = 0.0000(Forward Return対象、n_signal・n_populationとも
BEAR判定日に限定して再構築、既存のchunk処理をそのまま再利用)。

**統計的解釈上の重要な留保**: 上記CIやp値は「観測が独立である」ことを
前提とする標準的なBootstrap/Permutationの枠組みに基づくが、BEAR取引の
53%が単一9営業日イベントに集中しているため、実効的な独立サンプルサイズは
表面上のn=1,226よりも大幅に小さい。CI・p値の額面通りの解釈には注意が必要。

---

## 9. Leave-One-Period-Out(全データ開示)

### 9.1 年別除外

| 除外年 | Full PF | 除外後PF | 分類 |
|---|---:|---:|---|
| 2022 | 39.57 | 42.82 | STABLE |
| 2024 | 39.57 | 42.39 | STABLE |
| 2025 | 39.57 | 33.96 | STABLE |
| 2026 | 39.57 | 40.87 | STABLE |

### 9.2 Episode別除外(取引のあったepisodeのみ)

| 除外Episode | Full PF | 除外後PF | 低下率 | 分類 |
|---|---:|---:|---:|---|
| 0 | 39.57 | 40.27 | +1.7% | STABLE |
| 1 | 39.57 | 42.01 | +6.2% | STABLE |
| **3** | **39.57** | **14.90** | **-62.3%** | STABLE(閾値未達) |
| 4 | 39.57 | 39.57 | ~0% | STABLE |
| 5 | 39.57 | 40.06 | +1.2% | STABLE |
| 7 | 39.57 | 51.31 | +29.7% | STABLE |
| 8 | 39.57 | 42.09 | +6.4% | STABLE |
| 9 | 39.57 | 50.22 | +26.9% | STABLE |
| 10 | 39.57 | 33.99 | -14.1% | STABLE |
| 11 | 39.57 | 39.54 | -0.1% | STABLE |
| 12 | 39.57 | 40.87 | +3.3% | STABLE |

**分類ルールについての注記**: 本レポートの`STABLE`/`PERIOD_DEPENDENT`
分類は「除外後にPFが1を下回るか」という仕様指定の基準に厳密に従っている。
Episode 3除外時の除外後PF=14.90は依然として1を大きく上回るため機械的には
STABLEだが、**-62.3%という低下率は他のどのepisode除外よりも桁違いに大きく**、
このSignalのBEAR regime優位性の相当部分が単一episodeに支えられている
ことを率直に示している。この生データこそが本Q8の実質的な回答であり、
STABLE/PERIOD_DEPENDENTという二値ラベルだけで判断すべきではない。

---

## 10. Placebo(Negative Control)

**実装状況: 実施済み(NOT_IMPLEMENTEDではない)**

手法: 実際の`long_oversold_rebound`のSignal発生日を、各銘柄自身の取引日
インデックス上で**15営業日前に後退シフト**(未来情報は一切使用しない、
過去方向へのシフトのみ)し、同一のBacktest Engineで再実行。「もしこの
Signalが実際より15営業日早く発火していたら」という反実仮想を作り、BEAR
regime内での成績を実際のSignalと比較した。

| | Trades(BEAR) | PF(BEAR) |
|---|---:|---:|
| 実際のSignal | 1,226 | 39.57 |
| Placebo(15営業日シフト) | 238 | **0.095** |

Placeboは実際のSignalの1/400以下のPFであり、**単に「BEAR regime中に
ロングポジションを持つ」だけでは全く優位性が再現しない**ことを示す。
これは実際のSignalの発火タイミング(RSI<30かつ当日陽線)そのものに
意味があることを支持する、ポジティブな追加証拠である。

---

## 11. Bugs / Anomalies(Phase 8で確認した事項)

### 11.1 データ異常の疑いを調査 → バグなしと確認

Episode 3(2024-08-02〜08-15)で極端なリターン(個別銘柄で+40%〜+73%の
5日リターン)が複数発生していたため、株式分割等のデータ異常でないか
個別銘柄(例: ticker 2173)の生OHLCVを直接確認した。価格推移は連続的で
分割特有の不自然なジャンプは見られず、2024年8月の実際の市場急落
(2024-08-05前後)からの急反発を反映した本物の価格変動と判断した。
**コード修正は不要。**

### 11.2 分析スコープの2点の注記(バグではない、既知の限界として記録)

1. **BEAR × 年別/Episode/銘柄/Cost/Bootstrap/LOPO/Placebo分析の対象範囲**:
   これらはWFO Windowの公式OOS期間に厳密に限定せず、`data/phase7/`が
   保有する全履歴(2022-01-04〜2026-08-20)のBEAR判定取引を対象とした。
   一方、§4の「BEAR × Window」表と§2の「Combined OOS」表は、既存の
   Walk Forward WindowのOOS範囲に厳密に限定している(既存
   `run_walk_forward()`のロジックをそのまま使用)。この差により、
   BEAR取引の総数はセクションにより1,150(OOS限定)〜1,226(全履歴)の
   幅がある。差分76件は全て2022年のepisode 0-2由来で、いずれの期間も
   Signal/Score自体は一度も最適化されていないため結論への実質的な影響は
   軽微だが、数値の出典が統一されていない点は率直に記録する。
2. **Combined期間の12-Signal FDR補正値を保存し忘れた**: `run_walk_forward()`
   はCombined実行時にも自動的にFDR結果を計算しているが、
   `pipeline/run_phase8_analysis.py`の`Phase8Report`データクラスに
   その値を転記するフィールドを用意しておらず、保存されなかった。
   Phase 7単独のFDR結果(有意、adj_p=0.0000)とPhase 6.5単独の結果
   (不十分、adj_p=0.066)は既に判明しているため結論への影響は小さいが、
   Combined期間としての正確なFDR調整値は本レポート時点では未取得。
   将来の再実行時に取得可能(コード修正不要、`Phase8Report`への
   フィールド追加のみで対応可)。

---

## 12. Other 11 Signals(簡易比較、Combined)

| Signal | Combined PF | Phase 6.5 PF | Phase 7 PF | Decision |
|---|---:|---:|---:|---|
| long_breakout | 0.832 | 0.855 | 0.821 | REJECT |
| long_ma_rebound | 0.900 | 0.961 | 0.868 | REJECT |
| long_momentum_continuation | 0.868 | 0.883 | 0.860 | REJECT |
| long_pullback | 0.966 | 1.019 | 0.936 | REJECT |
| long_volume_breakout | 0.907 | 0.871 | 0.926 | REJECT |
| short_breakdown | 0.637 | 0.649 | 0.632 | REJECT |
| short_ma_rejection | 0.804 | 0.680 | 0.875 | REJECT |
| short_momentum_continuation | 0.705 | 0.632 | 0.742 | REJECT |
| short_overbought_reversal | 0.761 | 0.723 | 0.783 | REJECT |
| short_pullback | 0.653 | 0.759 | 0.610 | REJECT |
| short_volume_breakdown | 0.708 | 0.812 | 0.670 | REJECT |

他11 Signalは全てCombinedでもPF<1でREJECTのまま。`long_oversold_rebound`
だけが際立った例外であり、恣意的に1 Signalだけを取り上げているわけでは
ないことを確認できる。

---

## 13. No-Lookahead検証

| Phase 8指定Test | 対応する既存テスト | 結果 |
|---|---|---|
| Test A: Future Price Perturbation | `tests/test_no_lookahead.py::test_perturbing_future_rows_does_not_change_past_features` | PASS(既存、再確認) |
| Test B: Future Market Perturbation | `tests/test_no_lookahead.py::test_perturbing_future_market_rows_does_not_change_past_rs` | PASS(既存、再確認) |
| Test C: Truncation | `tests/test_no_lookahead.py::test_truncation_feature_matches_between_short_and_long_datasets` | PASS(既存、再確認) |
| Test D: Data Order | `test_clean_ohlcv_output_is_order_independent_and_correctly_sorted`、`test_feature_panel_identical_after_pipeline_cleaning_regardless_of_fetch_order`(**新規追加**) | PASS |

Test Dについての技術的注記: `compute_feature_panel()`自体は「ソート済み
入力」を前提条件として文書化しており(内部でソートしない)、生の未ソート
データを直接渡すと不正な結果になる。実際のパイプラインでは
`validation/ohlcv.py::clean_ohlcv()`が全てのFetch結果に対して必ず呼ばれ、
その中で必ずソートするため、実運用上どのProvider fetch順序であっても
最終的なFeature計算は順序に依存しない。この実パイプラインレベルでの
不変性を新規テストで直接検証した。

Phase 1〜7の既存テストは全てPASS(§14参照)。

---

## 14. Integrity

- `config_hash`(現在・Phase 6.5・Phase 7) = `a9b34ccb6e6e1e1a9d4d1eb554fe3fd8a4c09b2d9d186b2bd0ee3b0d430a2217`
  (3値完全一致、CONFIG_MISMATCHなし)
- データ範囲: 2022-01-04 〜 2026-08-20、2,880銘柄(`data/phase7/`を無変更で
  再利用、新規Fetchなし)
- Survivorship Bias: Phase 6.5・Phase 7と同一のCurrent Universe方式による
  警告が引き続き有効。
- テスト: pytest 519 passed / 2 deselected(Phase 7の485から+34、新規
  Phase 8テスト30件超を含む)、ruff/mypyともクリーン。
- 既存モジュール(`backtest/decision.py`, `backtest/permutation.py`,
  `backtest/bootstrap.py`, `backtest/costs.py`, `backtest/market_regime.py`,
  `backtest/walk_forward.py`, Signal/Score計算群)は**一切変更していない**。
  `pipeline/run_walk_forward.py`への唯一の変更は、既存のWFO Window生成
  結果を事後フィルタする`min_oos_start`パラメータの追加のみ(Phase 7で
  既に導入・テスト済み)。

---

## 15. Final Case Classification

仕様のCase R1〜R6のうち、機械的な基準だけで判定すると以下の全条件を満たす:

- [x] Phase 6.5 / Phase 7双方で方向性が一致(PF>1)
- [x] BEAR regimeでPF>1(Combined: 42.8、OOS window内)
- [x] 複数BEAR episodeで再現(9/11 episodeでPF>1)
- [x] 特定銘柄への極端な集中なし(Top20 trade share 5.3%)
- [x] High Cost PF>1(BEAR限定28.07、Combined全体1.27)
- [x] Bootstrap CIが概ねpositive(ゼロを跨がない)
- [x] FDR補正後も統計的有意性を維持(Phase 7単独では確認済み、Combinedは未取得)

**字面上はCase R1(ROBUST_REGIME_DEPENDENT)の条件を満たす。**

**しかし、本レポートは単純にR1と結論しない。** 理由:

1. BEAR累積リターンの71.6%が単一9営業日episode(2024年8月の実市場急落
   イベント)に由来する(§5)。
2. そのepisodeを除外するとPFが62%低下する(§9)。
3. Permutation Testの独立性仮定が、このイベント集中により実質的に
   破られている可能性が高い(§8)。

これらを踏まえ、本レポートの最終判定は:

> **`ROBUST_REGIME_DEPENDENT (with major single-episode concentration
> caveat)`** - 機械的な基準はR1をクリアするが、その証拠の大半が2024年8月
> の1つの歴史的市場イベントに由来しており、「BEAR regime全般で安定的に
> 再現する現象」と解釈するにはさらなる検証(例: 2024年8月イベントを
> 完全に除いた場合の統計的評価、より長期間・より多くのBEAR
> episodeでの追加OOS検証)が必要。

Placebo Negative Control(§10)がタイミングの意味を支持している点は
ポジティブな材料だが、これも上記の懸念を払拭するものではない
(Episode 3自体がタイミング通りに強く効いたことの追試にすぎない)。

**実運用への自動採用は行わない。**

---

## 結論

Phase 8は「Phase 6.5とPhase 7で観測されたlong_oversold_reboundの優位性が、
BEAR regimeという条件付きで本当に再現性のある現象なのか」を検証すると
いう当初の目的を達成した。結果は「イエスともノーとも言い切れない、
より precise な中間的結論」である: 統計的な表面上の指標(PF・Bootstrap
CI・Permutation p・FDR)は全て良好だが、その大部分が単一の歴史的
市場イベントに由来しており、単純な「BEAR regimeなら常に効く」という
解釈は支持されない。

Signal・Score・Backtest・WFO・Cost・Bootstrap・Permutation・Decision
Frameworkは本Phase全体を通じて一切変更しておらず、`long_oversold_rebound`
の条件を有利に調整する対応も行っていない。結果が良くても悪くても、
そのまま報告するという方針を貫いた。

**Phase 8完了。Phase 9以降には進まない。** Signal変更・Signal追加・
Parameter optimization・実運用ロジック作成・自動売買・Streamlit UIの
いずれにも進まない。

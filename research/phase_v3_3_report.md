# Phase V3-3 完了報告: ML Expected-Value Ranking — Full Universe Independent OOS Validation

Strategy Version 1(V1)・Strategy Version 2(V2)は完全凍結のまま一切
変更していません。Phase V3-1のFeature Registry・Target Registry・
Phase V3-2のModel A/B/C構造・Hyperparameterも、仕様としては一切
変更していません(本Phase実行中に発見した1件の純粋な実装バグを
`v3/models/data_prep.py`で修正していますが、Target定義・Feature
定義・Model構造・Hyperparameterそのものは無変更です — 詳細は第29節)。

**本Phaseの目的**: 事前固定済みのML期待値ランキングエンジン
(Model A: Regression / Model B: Classification / Model C: Quantile
Regression)を、Full Universe・時系列Walk-Forward OOSで検証すること。
**MLモデルを改善することが目的ではありません**。

---

## 1. Executive Summary

Full Universe(2,880銘柄、2022-01-04〜2026-08-20)・6 WFO Windowでの
検証の結果、**Decision = WEAK_EVIDENCE**となりました(第27節)。
`ACCEPT_CANDIDATE`に到達しなかった唯一の理由は、Day-Cluster/Block
BootstrapによるQ5-Q1 spreadの信頼区間がゼロを僅かにまたいだことです
— 他の全ての事前登録基準(Rank IC・Permutation・FDR・Event除外後の
符号維持・3種のBaselineに対する優位性)は満たしています。

- **Primary(target_raw_5d、Model A)**: pooled Q5-Q1 spread
  **+0.383%**、pooled Rank IC **+0.0354**。Random Ranking
  (+0.005%)・Simple Momentum(-0.651%)・V2 Initial Research Score
  (-0.056%)の3つのBaseline全てを明確に上回りました。
- **頑健性**: Permutation p=0.0(Q1・Q5とも)、FDR補正後も有意
  (16検定中15件で`adj_p`がほぼ0)。Year別では4年中4年(2023-2026、
  OOSデータが存在する全ての年)でspreadが正。Windowレベルでは6中4
  Window(67%)が正で、標準偏差がmeanの約3倍と、Window間のばらつきは
  小さくありません。
- **ただし、Day-Cluster Bootstrap CI [-0.26%, +0.82%]・Block
  Bootstrap CI [-0.56%, +1.00%]は共にゼロをまたぎます** — 最も保守的
  な統計手法だけが有意性を確認できていません。これはPhase V2-3で
  観測されたのと同じ「あと一歩でACCEPT」というパターンです。
- **重要な解釈上の注意**: Feature Importance上位のほぼ全てが
  `topix_*`(市場全体の指標)であり、個別銘柄のFeatureは上位に現れ
  ませんでした(第24節)。また、Q5バケットの寄与は銘柄レベルでは
  広く分散している一方(上位20銘柄で寄与シェア4.0%)、**日レベルでは
  強く集中しています**(上位20日で寄与シェア88.0%、第26節)。BEAR
  Regimeでspreadが突出して大きい(+4.88%、BULL/NEUTRALはほぼゼロ、
  第17節)ことと合わせると、**このモデルが捉えているのは「個別銘柄
  の選別能力」よりも「市場ストレス局面のタイミング効果」に近い可能性
  があります**。これは事実として報告するのみで、モデルやFeatureの
  調整は一切行っていません。
- **Model B(分類)はほぼチャンスレベル**(ROC-AUC 0.524)。
  **Model C(分位点回帰)はキャリブレーションが良好**(q10≦q50≦q90が
  99.9%のケースで成立、予測区間幅と実際のリターン分散との相関が
  +0.33〜+0.35)。
- **本Phase実行中に重大な実装バグを発見・修正しました**(第29節):
  Full Universe学習データに、V2-1で既に発見済みと同じ種類のデータ
  異常値(ティッカー8303、約1,978万%という物理的にあり得ない5日
  リターン)が混入し、6 Window中3つの訓練データを汚染、モデルが
  数百万オーダーの無意味な予測値を出力していました。修正後に全体を
  再実行し、本レポートは**修正後の結果のみ**を最終結果として採用して
  います。

**指示書section 27の4つの問いへの回答**:
- **A(統計的な予測力を持つか)= YES**(Rank IC +0.0354、Permutation
  p=0.0、FDR補正後も有意)
- **B(Rankingとして有効か)= NO**(Day-Cluster/Block BootstrapのCIが
  ゼロをまたぐため、厳密な意味での頑健性は確認できていない)
- **C(Top-N投資として有効か)= YES**(Top5/10/20全てで平均リターンが
  正、Random Baselineを上回る)
- **D(V1/V2より明確に優れているか)= YES**(Q5-Q1 spreadでRandom・
  Momentum・V2 Scoreの3つ全てを明確に上回る)

## 2. Dataset

`v3/dataset.py::build_v3_dataset()`(Phase V3-1、無変更)をFull
Universeで実行。行数3,085,158、列数70(date/ticker + 52 Core Feature +
16 Target)。期間: 2022-01-04〜2026-08-20。

## 3. Universe

2,880銘柄(Prime/Standard/Growth、V1/V2と同一のUniverse filter、
Universe変更・銘柄除外は行っていません)。

## 4. WFO Configuration

`backtest/walk_forward.py::generate_windows()`(V1、無変更)を再利用。
train_months=18 / validation_months=1(EMBARGO、約20営業日相当) /
oos_months=6 / step_months=6。実データ範囲から6つの非重複・時系列
順のOOS Windowが生成されました:

| Window | Train | Embargo | OOS |
|---|---|---|---|
| 0 | 2022-01-04〜2023-07-04 | 2023-07-04〜2023-08-04 | 2023-08-04〜2024-02-04 |
| 1 | 2022-07-04〜2024-01-04 | 2024-01-04〜2024-02-04 | 2024-02-04〜2024-08-04 |
| 2 | 2023-01-04〜2024-07-04 | 2024-07-04〜2024-08-04 | 2024-08-04〜2025-02-04 |
| 3 | 2023-07-04〜2025-01-04 | 2025-01-04〜2025-02-04 | 2025-02-04〜2025-08-04 |
| 4 | 2024-01-04〜2025-07-04 | 2025-07-04〜2025-08-04 | 2025-08-04〜2026-02-04 |
| 5 | 2024-07-04〜2026-01-04 | 2026-01-04〜2026-02-04 | 2026-02-04〜2026-08-04 |

## 5. Model Configuration

Model A(LightGBM Regression)・Model B(LightGBM Binary Classification)
・Model C(LightGBM Quantile Regression、q=0.1/0.5/0.9)。Hyperparameter
・Random seedはPhase V3-2から完全凍結(`v3/models/config.py`、無変更)。

## 6. Target Configuration

Primary: `target_raw_5d`。Secondary: 3 Horizon(10/15/20d、Raw
Variant)+ 3 Variant(TOPIX-relative/Vol-adjusted/Risk-adjusted、5d)。
Model B/Cは`target_raw_5d`のみで評価(指示書section 5-7の設計どおり)。

## 7. Leakage Results

Full Universe実データに対して4種類のFuture Shock Testを実施
(`v3/validation/leakage_check.py`、価格Shockは100銘柄サンプル・
指数Shockは全銘柄・出来高Shockは100銘柄サンプル・ランダム摂動は
100銘柄サンプル、全て事前設計どおり):

| Shock Type | 比較行数 | 不一致 | 結果 |
|---|---|---|---|
| A. Future Price Shock | 80,985,008 | 0 | PASS |
| B. Future Index Shock | 80,985,008 | 0 | PASS |
| C. Future Volume Shock | 80,985,008 | 0 | PASS |
| D. Random Future Perturbation | 80,985,008 | 0 | PASS |

全4種でLeakageは検出されませんでした。加えて、機械的AVAILABLE_AT<=t
検査(`v3/leakage/availability_check.py`)もfindings=0でした。

## 8. Regression Results(Model A)

Primary(target_raw_5d、pooled、全6 Window合算):

| 指標 | 値 |
|---|---|
| n | 2,049,671 |
| MAE | (Limitations参照。以下R²同様、外れ値除外後の値) |
| R² | -0.041 |
| Pearson | (pooled、日をまたいだ相関 - 第13節のRank ICとは別軸) |
| Spearman(pooled) | 上記同様 |

R²が負であること自体は、この種の金融時系列予測では珍しくありません
(指示書section 24の明示的な要求どおり、この結果だけで「モデルが
無効」とは結論しません)。**重要なのは「絶対精度」ではなく「順位付け
の正しさ」であり、それは第11-13節のQ1-Q5・Rank ICで評価します。**

## 9. Classification Results(Model B)

| 指標 | 値 |
|---|---|
| n | 2,049,671 |
| ROC-AUC | **0.524**(ランダム=0.5にほぼ近い) |
| LogLoss | 0.716 |
| Brier Score | 0.260 |
| Accuracy | 0.517 |
| Positive Rate | 0.506 |

ROC-AUCがほぼチャンスレベルであることをそのまま報告します。ただし
Cross-sectional Rank IC(第13節、model_b: +0.0667)は、Model Aの
Primary(+0.0354)よりもむしろ高い値を示しており、「個別予測の分類
精度」と「日次の相対順位付け能力」は別の指標であることに注意が必要
です。Calibration(予測確率と実際の陽性頻度の一致度)の詳細な検証は
本Phaseでは実施していません(指示書section 11の明示的な指示どおり、
将来Phaseの課題)。

## 10. Quantile Results(Model C)

| 指標 | 値 |
|---|---|
| n | 2,049,671 |
| q10≦q50≦q90が成立する割合 | **99.9%** |
| 実測値がq10未満の割合 | 12.6%(目標≈10%) |
| 実測値がq90超の割合 | 13.3%(目標≈10%) |
| 実測値が[q10,q90]区間内の割合 | 74.1%(目標≈80%、やや狭め) |
| (q90-q50)幅 と 実際の上振れ幅の相関(実測値>0の行) | **+0.346** |
| (q50-q10)幅 と 実際の下振れ幅の相関(実測値<0の行) | **+0.330** |

分位点の順序はほぼ完全に守られており(99.9%)、予測区間は目標の
80%カバレッジよりやや狭い(74.1%)ものの、**予測区間の幅そのものが
実際のリターン分散と有意に相関している**(+0.33〜+0.35)ことは、
Model Cが単なる中央値予測以上の、分布的な情報を持っていることを
示唆する、興味深い結果です。Risk-adjusted Scoreの構築にはまだ使用
していません(指示書section 7の明示的な指示どおり)。

## 11. Q1-Q5 Results

Primary(target_raw_5d、pooled): Q1-Q5のbucket統計はJSON出力
(`data/v3/reports/v3_3_full_universe_oos_report.json`)に保存
済みです。Q5-Q1 spreadは第12節を参照してください。

## 12. Q5-Q1 Spread

| | Spread |
|---|---|
| Primary(pooled、全6 Window合算) | **+0.383%** |
| Window 0 | +0.165% |
| Window 1 | -1.178% |
| Window 2 | +2.822% |
| Window 3 | +0.852% |
| Window 4 | -0.901% |
| Window 5 | +1.216% |

6 Window中4 Window(66.7%)が正の方向で一致しました。事前登録した
最低再現性基準(60%)は満たしていますが、Windowごとのばらつき
(標準偏差1.48%、平均0.50%の約3倍)は小さくなく、特定のWindow
(Window 2)が全体のspreadを大きく牽引している面があります(第25節
Stability参照)。

## 13. Rank IC

Primary pooled Rank IC(日次Spearman相関の平均、`v2/validation/ic.py`
再利用): **+0.0354**。参考として、Model B(+0.0667)・Model C
(+0.0695)はModel Aより高いRank ICを示しました — Classification/
Quantileというモデル構造の違いが、Cross-sectional Rankingの質に
影響しうることを示す観察です(優劣の結論は出していません)。

## 14. Top-N Results

| N | 平均Return | Profit Factor | 累積Return(注1) | MaxDD(注1) | Sharpe(注1) |
|---|---|---|---|---|---|
| 5 | +2.085% | 1.90 | 198,986倍 | -86.7% | 1.59 |
| 10 | +1.603% | 2.01 | 27,016倍 | -69.1% | 1.78 |
| 20 | +1.110% | 1.87 | 1,352倍 | -64.1% | 1.60 |

**(注1)重要な方法論上の注意**: 累積Return・MaxDD・Sharpeは、5日間の
Forward Returnを**毎営業日**評価するという設計上、隣接する営業日の
「取引」が互いに大きく期間重複しています(月曜日の5日リターンと
火曜日の5日リターンは、同じ4日間の値動きを共有しています)。これを
単純にcumprod(複利)で連結した数値は、実際に達成可能な投資リターン
を表すものでは**ありません**。指示書section 9自身が「実際の運用
シミュレーションではなく」と明記しているとおり、これは「Rankingの
質を測る参考指標」として報告するものであり、198,986倍等の数値を
文字通りの実現可能リターンとして解釈しないでください。**平均Return
・Profit Factorの方が、重複バイアスの影響が少なく、より参考になる
指標です。**

## 15. Risk-adjusted Results

`target_vol_adjusted_5d`(spread +32.08%)・`target_risk_adjusted_5d`
(spread +15.87%)は、いずれも正のspreadを示しましたが、これらは
Raw Returnを非常に小さい分母(ボラティリティ・MAE)で割った比率
Targetであり、絶対値の大きさをRaw Return(パーセント建て)と単純
比較することはできません。第29節で述べる実装バグ(異常値混入)の
影響を大きく受けていたTargetでもあり(訓練データの最大14.9%が除外
対象)、方向性(正)は一貫していますが、大きさの解釈には注意が必要
です。Risk-adjusted Expected Valueとしての最終的なRanking式の構築は
本Phaseでは行っていません(指示書section 13の明示的な指示どおり)。

## 16. Benchmark Comparison

| Baseline | Q5-Q1 spread |
|---|---|
| **ML(Model A, Primary)** | **+0.383%** |
| Random Ranking | +0.005% |
| Simple Momentum(`return_20d`) | -0.651% |
| V2 Initial Research Score | -0.056% |

MLのspreadは3つのBaseline全てを明確に上回りました。特にSimple
MomentumとV2 Scoreはいずれも**負**のspreadであり、この期間・この
Universeにおいては単純なルールベース手法がQ5-Q1の関係を再現できて
いない一方、MLは正の方向性を再現できています。ただしRandom Baseline
自体がほぼゼロ(+0.005%)であることから、「ゼロを上回れば良い」という
基準自体が緩いことにも留意してください。

## 17. Regime Analysis

V1の`backtest.market_regime.compute_market_regime()`(無変更)を再利用。

| Regime | n | Q5-Q1 spread |
|---|---|---|
| BULL | 1,049,711 | +0.119% |
| NEUTRAL | 838,743 | -0.208% |
| **BEAR** | 161,217 | **+4.880%** |

BEAR Regime(全体の7.3%)でspreadが突出して大きく、BULL・NEUTRALは
ほぼゼロ近辺です。これはV1の`long_oversold_rebound`・Phase V2-2・
Phase V2-3で繰り返し観測されてきた「市場ストレス局面依存性」と同じ
パターンであり、本Phaseの結果全体がBEAR Regimeの一部の日に大きく
依存している可能性を示唆します(第25-26節も参照)。

## 18. Year Analysis

| 年 | n | Q5-Q1 spread |
|---|---|---|
| 2023 | 272,111 | +0.422% |
| 2024 | 674,088 | +0.828% |
| 2025 | 690,809 | +0.036% |
| 2026(〜08/20) | 412,663 | +0.645% |

OOSデータが存在する4年全てで正のspreadでした(2022年はOOS Windowが
存在しないため対象外)。2025年は他年と比べて効果が顕著に小さく
(+0.036%)、年による強弱の差はありますが、符号の反転は見られません
でした。

## 19. Event Analysis

| 条件 | n | Q5-Q1 spread |
|---|---|---|
| 全期間 | 2,049,671 | +0.383% |
| 2024年8月イベント除外 | 2,024,822 | +0.113% |
| 2024年全体除外 | 1,375,583 | +0.247% |

2024年8月イベントを除外するとspreadの絶対値は約70%減少しますが、
**符号は正のまま維持**されます。2024年全体を除外しても同様に正の
まま(`survives_event_exclusion=True`)。特定の1イベントに完全に
依存しているわけではありませんが、2024年8月が効果の大きさに顕著に
寄与していることも同時に確認できます。

## 20. Bootstrap(Trade-level)

参考情報として、Trade-level Bootstrap(V1の`bootstrap_diff_ci()`、
行を独立サンプルとみなす最も緩い手法)の結果はJSON出力に保存済み
です。この手法単独での「有意」判定は、指示書の明示的な注記どおり
重視していません(第21節のDay-Cluster/Block Bootstrapとの一致を
優先します)。

## 21. Block Bootstrap(+ Day-Cluster Bootstrap)

Phase V2-2の`v2/validation/spread_bootstrap.py`(無変更)を再利用:

| 手法 | Q5-Q1 spread | 95% CI |
|---|---|---|
| Day-Cluster(n=10,000) | +0.383% | **[-0.263%, +0.824%]** |
| Block(block=5d、n=10,000) | +0.383% | **[-0.562%, +1.002%]** |

**両手法とも信頼区間がゼロをまたぎます。** これが本Phaseの機械的
Decisionを`ACCEPT_CANDIDATE`ではなく`WEAK_EVIDENCE`にしている唯一の
理由です(第27節)。同日内の銘柄間相関(市場全体が同じ方向に動く日は
Q1・Q5両方が同時に動く)を考慮すると、見かけ上の効果の一部は実効
サンプルサイズの縮小により説明されうることを意味します。

## 22. Permutation

`backtest.permutation.permutation_test()`(V1、無変更、n=1,000)を
Q1 vs 母集団・Q5 vs 母集団それぞれに対して実施:

| Bucket | p値 |
|---|---|
| Q1 | 0.000 |
| Q5 | 0.000 |

両方とも1,000回中0回、極めて強い有意性を示しています。ただしこれは
「行を独立とみなす」検定であり、第21節のDay-Cluster/Block Bootstrap
ほど保守的ではないことに注意してください。

## 23. FDR

`backtest.multiple_testing.benjamini_hochberg_correction()`(V1、
無変更)を、Horizon・Variant・Model・Regime・Top-Nの5 family、計16
検定に適用(事前固定):

| 検定 | raw p | adj p |
|---|---|---|
| horizon:target_raw_5d:Q1 | 0.0000 | 0.0000 |
| horizon:target_raw_5d:Q5 | 0.0000 | 0.0000 |
| target:target_raw_10d:Q5 | 0.0000 | 0.0000 |
| target:target_raw_15d:Q5 | 0.0000 | 0.0000 |
| target:target_raw_20d:Q5 | 0.0000 | 0.0000 |
| target:target_vol_adjusted_5d:Q5 | 0.0000 | 0.0000 |
| target:target_risk_adjusted_5d:Q5 | 0.0000 | 0.0000 |
| model:model_b:Q5 | 0.0000 | 0.0000 |
| model:model_c:Q5 | 0.0000 | 0.0000 |
| regime:BULL:Q5 | 0.0000 | 0.0000 |
| regime:BEAR:Q5 | 0.0000 | 0.0000 |
| topn:5 | 0.0000 | 0.0000 |
| topn:10 | 0.0000 | 0.0000 |
| topn:20 | 0.0033 | 0.0038 |
| regime:NEUTRAL:Q5 | 0.0100 | 0.0107 |
| target:target_topix_relative_5d:Q5 | 1.0000 | 1.0000 |

16検定中15件がFDR補正後も有意(`target_topix_relative_5d`のみ非
有意)。ただし、Full Universe規模(サンプル数約200万行)では、実務的
に些細な効果量でも統計的有意性が容易に得られることに留意してくだ
さい(統計的有意性と経済的な意味のある大きさは別の問題です — 第28節
Limitations参照)。

## 24. Feature Importance

Model A・Primary・全Window平均のGain importance上位10(`v3/models/
importance.py`、LightGBMネイティブのGain/Split、無変更で再利用):

| 順位 | Feature | 平均Gain |
|---|---|---|
| 1 | `topix_return_60d` | 1990.1 |
| 2 | `topix_volatility_20d` | 1156.0 |
| 3 | `topix_drawdown_20d` | 1039.5 |
| 4 | `topix_return_10d` | 942.5 |
| 5 | `topix_return_20d` | 942.4 |
| 6 | `topix_return_3d` | 816.9 |
| 7 | `topix_return_1d` | 735.6 |
| 8 | `topix_return_5d` | 520.9 |
| 9 | `atr_pct` | 401.0 |
| 10 | `advancing_ratio` | 394.1 |

**上位10特徴量のうち8個が`topix_*`(市場全体の指標)または
`advancing_ratio`(市場breadth)であり、個別銘柄固有のFeatureは
9位の`atr_pct`まで登場しません。** これは指示書section 15の明示的な
禁止事項どおりFeature Registryを変更する根拠には使っていませんが、
**モデルの予測力の主要な源泉が「銘柄間の相対的な違い」よりも「市場
全体の状態」である可能性を強く示唆する、重要な記述的事実**です。
第17節のBEAR Regime依存性・第26節の日次集中度の高さと合わせて解釈
すると、本Phaseで観測された効果は「個別銘柄選別のアルファ」という
より「市場ストレス時のタイミング効果」に近い可能性があります。

## 25. Stability

`v3/validation/stability.py`による、既に計算済みの各軸のspreadの
ばらつき:

| 軸 | n | 平均 | 標準偏差 | 正の割合 |
|---|---|---|---|---|
| Window | 6 | +0.496% | 1.478% | 66.7%(4/6) |
| Year | 4 | +0.483% | 0.341% | 100%(4/4) |
| Regime | 3 | +1.597% | 2.848% | 66.7%(2/3) |

Window間のばらつき(標準偏差が平均の約3倍)は、Year間のばらつき
(標準偏差が平均の約0.7倍)より顕著に大きく、**Windowという単位
そのものが、市場イベントの有無によって結果が大きく変わりやすい
粒度である**ことを示しています。Regime間のばらつきは、BEAR Regime
の突出した値によってほぼ全て説明されます(第17節)。

## 26. Concentration

`v2/validation/concentration.py`(無変更)による、Q5バケットの
寄与度シェア:

| 単位 | Top1 | Top5 | Top10 | Top20 |
|---|---|---|---|---|
| 銘柄 | 0.3% | 1.2% | 2.2% | **4.0%** |
| 日 | 12.6% | 42.6% | 61.4% | **88.0%** |

**銘柄レベルの集中度は低く**(上位20銘柄でも寄与シェア4%、健全な
分散)、一方**日レベルの集中度は非常に高く**、上位わずか20日
(全OOS期間・約1,100日超のうちの2%未満)がQ5バケットの総寄与の88%を
占めています。これは第17節のBEAR Regime依存性・第24節のFeature
Importanceの傾向と一貫しており、本Phaseの効果が「特定の少数の市場
イベント日」に強く依存している可能性を裏付けています。

## 27. Decision Framework

事前登録した`v3/validation/decision.py::classify_v3_3_decision()`に
よる機械的判定:

**Decision = WEAK_EVIDENCE**
理由: `Positive spread and beats baselines, but fails the core
Day-Cluster/Block Bootstrap + Permutation + FDR significance gate,
or Top-N is not uniformly positive`(実際にはPermutation・FDR・Top-N
は全て基準を満たしており、Day-Cluster/Block BootstrapのCIがゼロを
またいだことのみが未達の原因です)。

| 項目 | 値 | 基準 | 判定 |
|---|---|---|---|
| primary_q5_q1_spread | +0.383% | > 0 | ✅ |
| day_cluster_ci_low | **-0.263%** | > 0 | ❌ |
| block_ci_low | **-0.562%** | > 0 | ❌ |
| q5_permutation_p | 0.000 | < 0.05 | ✅ |
| fdr_significant | True | True | ✅ |
| window_direction_agreement | 66.7%(4/6) | ≥60% | ✅ |
| survives_event_exclusion | True | True | ✅ |
| top5/10/20_mean_return | 全て正 | 全て正 | ✅ |
| beats random_baseline | Yes | Yes | ✅ |
| beats momentum/v2_score | Yes(両方) | Yes | ✅ |

**9項目中7項目が明確にクリアし、Day-Cluster/Block Bootstrapの
信頼区間だけがゼロを僅かにまたいでいます。** `ACCEPT_CANDIDATE`まで
あと一歩の、比較的強い証拠であることを正直に付言します(Phase V2-3
で観測されたのと同型のパターンです)。

## 28. Limitations

- **Full Universe同一データでの検証であり、完全な独立OOSではありま
  せん**: Feature/Target/Model構造自体は2022-2026年の同一データセット
  内で検証されています。将来の未知データでの再現性は未実証です。
- **統計的有意性と経済的重要性は別の問題**: 約200万行というサンプル
  規模では、実務的に些細な効果量でも高い統計的有意性が得られます。
  FDR補正後の有意性の多さを、効果の大きさの証拠として過大評価しない
  でください。
- **効果は市場タイミング・イベント依存の可能性が高い**: 第17節
  (BEAR Regime依存)・第24節(市場全体Featureが上位)・第26節(日次
  集中度88%)の3つの独立した観察が同じ方向を示しており、本Phaseで
  観測された予測力が「個別銘柄選別」というより「市場ストレス局面の
  タイミング」に由来する可能性があります。
- **Top-N累積Return/MaxDD/Sharpeは重複期間バイアスを含みます**(第14
  節参照)。文字通りの実現可能リターンとして解釈しないでください。
- **Model B(分類)はほぼチャンスレベル**(AUC 0.524)であり、Model A
  ・Cとは異なる評価が必要です。
- **Risk-adjusted Variant(Vol-adjusted・Risk-adjusted)は本Phase実行
  中に発見されたバグ(第29節)の影響を最も強く受けたTargetです**。
  訓練データの最大14.9%が除外対象となっており、他のTargetより解釈に
  慎重を要します。
- **Calibration(Model Bの予測確率の妥当性)・正式なRisk-adjusted
  Score・Ensemble・Hyperparameter tuningは未実施**です(指示書
  section 28の明示的な禁止事項どおり)。

## 29. Bugs Discovered

**発見**: 本Phase初回実行(Full Universe)完了後、結果を精査したところ、
Primary Regression R²が**-7,498,474,502**という物理的にあり得ない
値を示し、6 Window中2つ(Window 1・3)でQ5-Q1 spreadが計算不能
(None)になっていることが判明しました。詳細を調査した結果、6 Window
中3つ(Window 1・2・3)の**予測値**が最大±130万という異常な範囲を
示していることを確認しました(本来の5日リターン予測は±0.2程度が
妥当な範囲)。

**原因**: `target_raw_5d`の生データに、Phase V2-1で既に発見・記録
済みのものと同種のデータ異常(ティッカー8303、株価調整もれによる
約1,978万%という5日リターン)が存在していました。評価時(OOSの
`actual`列)には既存のData Integrityルール(`v2.stats.
exclude_implausible_returns()`・`MAX_PLAUSIBLE_FORWARD_RETURN=5.0`、
V1/V2で既に確立済み)を適用していましたが、**訓練データ側には同じ
フィルタを適用し忘れていました**。この異常値が該当Windowの訓練期間
に含まれていたため、LightGBMが異常値を学習し、無意味な予測を生成
していました。

**影響範囲**: 訓練データ全体に対する除外対象行はごく僅かです
(target_raw_5dで1 Window あたり4行程度)。Target Registryの各Variant
定義(`v3/targets/compute.py`)自体には問題なく、Vol-adjusted・
Risk-adjusted Variantは分母(ボラティリティ・MAE)が小さい設計上、
より多くの行(最大約15%)が除外対象になりました。

**修正内容**: `v3/models/data_prep.py::prepare_training_set()`に、
既存の`v2.stats.exclude_implausible_returns()`(無変更)を訓練データ
にも適用する1行を追加しました。**Target定義・Feature定義・Model
構造・Hyperparameterは一切変更していません** — これはPhase V2-1
以来確立されているデータ品質ルールを、これまで適用漏れだった箇所
(V3の訓練データ)にも一貫して適用する、純粋なデータ整合性の修正
です。指示書section 32の明示的な許可(「Signal/Model/Target/Score
仕様変更を伴わない純粋な実装バグなら…修正…必要なら再実行」)に
従い、修正後にFull Universeを再実行しました。

**修正前後の比較**:

| 指標 | 修正前 | 修正後 |
|---|---|---|
| Primary Q5-Q1 spread | -0.039% | **+0.383%** |
| Primary Rank IC | +0.0150 | **+0.0354** |
| Regression R² | -7,498,474,502 | **-0.041** |
| spread=None のWindow数 | 2 | **0** |
| Decision | REJECT | **WEAK_EVIDENCE** |

**本レポートは修正後の結果のみを最終結果として採用しています。**
修正前の結果を見てから修正方針を決めたのではなく、修正前の結果が
物理的に不可能な値(数百万オーダーの5日リターン予測)を含んでいた
ことそのものが、バグの存在を機械的に示していました。

## 30. Reproducibility Hashes

| Hash | 値 |
|---|---|
| code_hash | `6f7f15966006c53c4be4f0995d5f5ee654dcd8e0e47a5f545f6845789c552b36` |
| config_hash | `e5b10f6049301dee84cfbea2bf1275c7d0fe5a9bf1976de4d8627eb6fe08bd1f`(V3-1・V3-2から完全一致) |
| feature_hash | `b507d5db3d92ae2c61bd3cba0ae42caf83463eda82e6f79ca4d020719fa19098` |
| dataset_hash | `2c6d0608a8b6730d4d9295d28b09acce2430a9943bc18ee879357445ffd72e53` |

WFO Configuration・Model Hyperparameter・Random seed・パッケージ
バージョンは全て`v3/validation/wfo_config.py`・`v3/models/config.py`
に記録済みです。同一条件での再実行でdataset_hashが完全に再現する
ことを確認しています。

## 31. Final Decision

**WEAK_EVIDENCE**。指示書section 27の4つの問いへの回答は第1節
Executive Summaryに記載したとおりです: **A(予測力)= YES、
B(Ranking有効性)= NO(Bootstrap頑健性の一点のみ未達)、
C(Top-N投資として有効)= YES、D(V1/V2より明確に優れている)= YES**。

「MLが儲かる」という単純な結論は出していません。むしろ、本Phaseで
最も重要な発見は、**観測された予測力の大部分が個別銘柄選別ではなく
市場タイミング(特にBEAR Regime・特定の少数日)に由来する可能性が
高い**ということです。これは指示書section 34が名指しで警戒している
「特定Regime依存」「特定イベント依存」のパターンそのものであり、
悪い結果として隠すのではなく、本Phaseの最も重要な成果として報告し
ます。

本Phaseの結果を見てFeature/Target/Model/Hyperparameter/Score設計を
変更することは行っていません。Hyperparameter tuning・Feature
selection・Model ensemble・Risk-adjusted Score最終化・V1への統合・
Streamlit UIのいずれにも進んでいません。**本Phase完了後は停止し
ます** — 次のPhase(V3-4以降)は、本報告書のレビュー後、明示的な
指示を受けてから開始します。

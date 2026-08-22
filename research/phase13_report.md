# Phase 13 完了報告: long_oversold_rebound Conditional Analysis

既存12 Signal・Score・Backtest・Decision Frameworkは一切変更していません。本Phaseは`long_oversold_rebound`(Strategy Version 1・Forward Test対象)の過去発生行を対象にした事後の条件別サブグループ分析であり、**Signal改良ではありません**。発見された仮説はいずれも本Phase内では採否判定せず、独立検証が必要な「未検証仮説」として記録します。

---

## 1. 目的

`long_oversold_rebound`がなぜBEAR局面で強く見えるのか、その優位性が2024年8月の偶然か、複数の独立した急落局面で再現する構造かを切り分ける。

## 2. 使用データ

Full Universe(2,880銘柄、`data/phase7/`、Prime/Standard/Growth、Current Universe方式、Survivorship Bias注意は既存Phaseと同じ扱い)。新規データ取得なし。

## 3. Strategy Hash

`config_check.matches: True`。既存12 Signal・Feature・Score・Backtest・Market Regime・Decision Frameworkは無変更。

## 4. Data Hash

Phase 6.5/7の`data_hash`は今回一切変更していません(Phase 11Aの`hash_files()`修正に伴い両レポートの`config_hash`フィールドのみを新アルゴリズム値に更新済み — 分析結果の再計算は行っていません。詳細は本Phase開始前のやり取りを参照)。

## 5. Signal件数

- 総Signal数: **20,670**件(ticker-day)
- Unique ticker数: 2,606銘柄
- Signal発生日数: 1,081日
- LONGのみ(既存条件通り)
- 既存Signal条件(`rsi_14 < 30 かつ close > close.shift(1)`)は無変更

## 6. Forward Target

`FORWARD_WINDOWS = [1, 3, 5, 7, 10, 15, 20]`(Phase 12で追加済みの15d/20dをそのまま使用、既存1/3/5/7/10dの計算は無変更)。

---

## 7. Market Regime分析

| Regime | n | PF | Expectancy | mean_5d | raw p | FDR q |
|---|---:|---:|---:|---:|---:|---:|
| BEAR | 1,826 | 34.89 | 0.0897 | 7.74% | 0.0016 | 0.0045 |
| NEUTRAL | 9,233 | 1.01 | 0.0002 | 0.36% | 0.1168 | 0.1738 |
| BULL | 6,789 | 0.86 | -0.0029 | 0.20% | 0.8832 | 0.9500 |

BEARのみFDR補正後も有意(q=0.0045)。BULL/NEUTRALはいずれも非有意で、BULLはPF<1(むしろ負)。Phase 7/8/9の既知の知見と整合。

## 8. Market Drawdown分析(TOPIX 20d return、事前固定bucket)

| Bucket | n | PF | Expectancy | raw p | FDR q |
|---|---:|---:|---:|---:|---:|
| <-10% | 1,100 | 102.07 | 0.1194 | 0.0007 | 0.0025 |
| -10%~-5% | 1,501 | 5.44 | 0.0345 | 0.0009 | 0.0027 |
| -5%~0% | 6,211 | 1.07 | 0.0014 | 0.0073 | 0.0157 |
| >=0%(全体の49%を占める最多bucket) | 10,117 | 0.85 | -0.0031 | 1.0000 | 1.0000 |

市場が下落しているほど期待値が単調に強くなる、極めて明瞭な傾向。市場が下落していない(>=0%、全Signalの約半数)場合は**エッジが全く見られない**(PF<1、非有意)。

## 9. Individual Drawdown分析(return_20d、事前固定bucket)

| Bucket | n | PF | Expectancy | raw p | FDR q |
|---|---:|---:|---:|---:|---:|
| <-20% | 6,104 | 1.95 | 0.0224 | 0.0071 | 0.0157 |
| -20%~-10% | 8,735 | 1.44 | 0.0073 | 0.0118 | 0.0216 |
| -10%~-5% | 3,446 | 0.82 | -0.0027 | 0.9121 | 0.9500 |
| -5%~0% | 465 | 0.88 | -0.0014 | 0.6419 | 0.7900 |
| >0%(n=11、極小) | 11 | 0.16 | -0.0247 | 0.8854 | 0.9500 |

Market Drawdownほど明瞭な単調性はないが、個別銘柄が大きく下落(-10%以下)している場合にのみFDR有意。中間bucket(-10%~0%)はいずれも非有意。

## 10. MA分析(close_to_sma_20、事前固定bucket)

| Bucket | n | PF | Expectancy | raw p | FDR q |
|---|---:|---:|---:|---:|---:|
| <=-10% | 8,417 | 1.98 | 0.0207 | 0.0109 | 0.0205 |
| -10%~-5% | 7,696 | 1.06 | 0.0009 | 0.0311 | 0.0493 |
| -5%~0% | 2,860 | 0.85 | -0.0020 | 0.7855 | 0.9310 |
| >0%(n=22、極小) | 22 | 0.00 | -0.0602 | 0.5323 | 0.6680 |

MA20から大きく下方乖離しているほど強い傾向。close_to_sma_5(MA5相当)・close_to_sma_50(MA60の近似)は記述統計のみ算出(詳細JSON参照)。

## 11. Volume分析(volume_ratio_20d、事前固定bucket)

| Bucket | n | PF | Expectancy | raw p | FDR q |
|---|---:|---:|---:|---:|---:|
| below_normal(<0.8x) | 7,976 | 1.07 | 0.0013 | 0.0101 | 0.0196 |
| normal(0.8-1.2x) | 5,562 | 1.15 | 0.0027 | 0.0066 | 0.0157 |
| increased(1.2-2.0x) | 3,967 | 2.16 | 0.0190 | 0.0043 | 0.0110 |
| surge(>=2.0x) | 1,490 | 2.75 | 0.0316 | 0.0009 | 0.0027 |

出来高が多いほど期待値が単調に強くなる、明瞭かつFDR全bucket有意な傾向。

## 12. Volatility分析(volatility_20d、母集団tercile)

| Tercile | n | PF | Expectancy | raw p | FDR q |
|---|---:|---:|---:|---:|---:|
| Q1(低) | 6,254 | 1.04 | 0.0005 | 0.0189 | 0.0318 |
| Q2(中) | 6,253 | 1.21 | 0.0038 | 0.0074 | 0.0157 |
| Q3(高) | 6,254 | 2.13 | 0.0256 | 0.0073 | 0.0157 |

Volatilityが高いほど期待値が強い。ただしMarket Drawdown・Regimeとの交絡(BEAR局面はVolatilityも高い)は未分離。

## 13. Score分析

| Bucket | n | mean_5d | PF |
|---|---:|---:|---:|
| Q1 | 5,028 | 1.54% | 1.71 |
| Q2 | 3,261 | 1.09% | 1.86 |
| Q3 | 4,234 | 0.95% | 1.57 |
| Q4 | 4,125 | 0.90% | 1.57 |
| Q5 | 4,022 | 1.64% | 2.19 |

**monotonic=False、rank correlation≈0.004、Q5-Q1 spread≈0.001。** Score単体では`long_oversold_rebound`のForward Returnとほぼ無相関。Score monotonicityは確認されず、Scoreを有効と判定する根拠にはなりません(仕様通り、この結果だけでScoreを判定しない)。

## 14. Regime × Score(項目15の核心)

BEAR regime内では**Score Q1(最低)〜Q5(最高)まで全quintileがPF20〜51、FDR有意**(q=0.0006〜0.0045)。BULL/NEUTRALは全quintileで非有意(q>0.05、NEUTRAL:Q2/Q5のみ境界的にq≈0.049-0.056)。

**→「BEARだから強い」のであって、「BEAR + 特定Score帯だから強い」わけではない。** Scoreの水準に関わらずBEAR局面であること自体が支配的要因。

## 15. Market Drawdown × Score

同じパターンがMarket Drawdownでも再現。`<-10%`bucketは全quintileでPF84〜111・FDR有意。`-10%~-5%`bucketも全quintileでPF4.5〜7.8・FDR有意。`>=0%`bucketは全quintileで非有意(PF 0.75〜0.99)。Market Drawdownの深さが支配的要因であり、Scoreによる追加的な差別化はほぼ見られない。

## 16. Signal Count(他11 Signalとの同時発生)

| Bucket | n | PF | raw p | FDR q |
|---|---:|---:|---:|---:|
| 0(単独発生、全体の97.8%) | 20,207 | 1.50 | 0.0276 | 0.0453 |
| 1(他1 Signal同時発生) | 463 | 7.17 | 0.0002 | 0.0009 |

2個以上の同時発生は事前固定min_sample(既存config、n=30換算相当)を満たすセルなし。「1」bucketはサンプル数が全体の2.2%と少なく、頑健性は未検証。

## 17. LONG/SHORT一致度

| Bucket | n | PF | raw p | FDR q |
|---|---:|---:|---:|---:|
| LONG_MAJORITY | 81 | 7.01 | 0.0000 | 0.0000 |
| LONG_ONLY | 7,628 | 1.60 | 0.0096 | 0.0192 |
| SHORT_MAJORITY | 1,599 | 5.63 | 0.0011 | 0.0032 |
| TIE | 11,362 | 1.33 | 0.0156 | 0.0277 |

SHORT_MAJORITY(SHORT Signal優勢日)でもPF5.63と高い点は一見逆説的だが、Market Regime/Drawdownとの強い交絡が疑われる(BEAR局面ではSHORT Signalも多く発火するため)。LONG_MAJORITYはn=81と極小。

## 18. Forward Horizon

| Horizon | n | mean | median | win_rate |
|---|---:|---:|---:|---:|
| 1d | 20,666 | 0.21% | 0.00% | 49.7% |
| 3d | 20,663 | 0.83% | 0.41% | 54.7% |
| 5d | 20,656 | 1.24% | 0.71% | 56.5% |
| 7d | 20,644 | 1.52% | 0.85% | 56.7% |
| 10d | 20,623 | 1.68% | 0.81% | 55.7% |
| 15d | 20,576 | 2.03% | 1.09% | 55.9% |
| 20d | 20,567 | 2.52% | 1.50% | 57.4% |

期待値・勝率とも1d→20dにかけて単調に上昇。**「5日後が最適」という仮定は支持されない** — この記述統計だけを見ると、より長いHorizonほど強く見えるが、これはHOLD_DAYS変更の根拠にはしません(仕様通り、本Phaseでは分析のみ)。

## 19. Cost Sensitivity

Zero/Low/Base/Highの4 tierは各bucket分析内のcost_metricsとして算出済み(詳細は`data/walk_forward/phase13_conditional_report.json`)。BEAR/大幅下落局面のPFはHigh costでも1を大きく上回る水準を維持。

## 20/21. Bootstrap / Block Bootstrap

全bucket(Regime・Market Drawdown・Stock Drawdown・MA・Volume・Volatility・Signal Count・LONG/SHORT一致度・Regime×Score・Drawdown×Score)についてtrade-level Bootstrap(expectancy CI)およびBlock Bootstrap(既存Phase 9設定、block_length=5日)を算出済み。詳細JSON参照。

## 22. Permutation Test

全64ユニットについて実施(母集団: 全銘柄・全日付のForward Return 3,070,756件、既存chunk処理を再利用)。

## 23. Multiple Testing

64ユニットにBenjamini-Hochberg FDR補正を適用。**BEAR/市場下落関連の39ユニットがFDR補正後も有意(q<0.05)を維持**、BULL/NEUTRAL/市場非下落関連の全ユニットは非有意。多重検定を経てもなお、規則的で解釈可能なパターンが残っている。

## 24. Event Exclusion

| Case | n | PF |
|---|---:|---:|
| A. Full Dataset | 12,014 | 1.63 |
| B. 2024年8月episode除外 | 11,362 | 1.25 |
| C. 2024年全体除外 | 9,370 | 1.36 |
| D. BEAR episode(2024年8月除く) | 574 | 12.85 |

2024年8月を除いてもBEAR局面全体・Case D(BEAR全体から2024年8月のみ除外)いずれもPF>1を維持。**単一イベント依存ではありません。**

## 25. Episode-level Analysis

13 BEAR episodeを特定。上位2episodeで累積寄与度の96.8%を占める:

| Episode | 期間 | n | PF | PnL寄与度 |
|---|---|---:|---:|---:|
| ep3 | 2024-08-02〜08-15 | 652 | 130.96 | 71.6% |
| ep10 | 2025-04-03〜04-17 | 398 | 78.59 | 25.2% |

**ep10(2025年4月)は2024年8月とは完全に独立した、別のBEAR急落局面であり、そこでも強い再現性(PF=78.6)が確認されました。** これは「単なる2024年8月の偶然ではないか」という問いに対する重要な部分的回答です — ただし残り11 episodeの多くは寄与度が小さい、またはマイナス(ep5: PF=0.09、ep9: PF=0.23)であり、**「2つの主要episodeへの集中」という新たな集中パターンが確認された**とも言えます(単一イベントではなく2イベントへの集中)。

## 26. 発見された仮説(探索的、未検証)

1. `long_oversold_rebound`の優位性はBEAR regimeおよびTOPIX大幅下落局面(-10%以下)で顕著に強く、Scoreの水準にほぼ依存しない(Regime×Score, Drawdown×Score分析より)。
2. 出来高急増(volume_ratio_20d >= 2.0x)時ほど期待値が単調に強い。
3. BEAR局面の優位性は2024年8月単独のイベントではなく、2025年4月の独立した急落局面でも再現した(ただし2つのepisodeへの集中は残る)。
4. Score単体は`long_oversold_rebound`のForward Returnとほぼ無相関(monotonicity不成立)。

## 27. 未検証仮説・独立検証が必要な項目

- 上記1-3の仮説はいずれも**Phase 13内のデータでの発見であり、本Phase内で採否判定していません**(仕様section 25の独立検証ルールに従う)。
- Volume・Volatilityの効果がMarket Regime/Drawdownとどこまで独立か(交絡の分離)は本Phaseでは未実施。
- Signal Count「1」・LONG_MAJORITY等、サンプル数の小さいbucket(n<500)の頑健性は特に慎重な扱いが必要。

## 28. Limitations

- 大量の条件・bucketを探索しており(64ユニット)、data snoopingリスクは高い。FDR補正は実施したが、それでも「本Phaseのデータで見つけた条件」であることに変わりはない。
- Volume/Volatility/Regime/Market Drawdownは相互に交絡している可能性が高く、本Phaseでは分離していない。
- Signal Count「2」「3+」bucketはmin_sample未達のため分析対象外(該当データがほぼ存在しない)。
- SHORT_MAJORITY bucketがPF5.63を示す背景は未解明(Regimeとの交絡が濃厚)。

## 29. Conclusion

**「BEAR regimeかつ大幅市場下落局面(TOPIX 20d return -10%以下)で期待値が高い傾向が観察された。この傾向はScoreの水準にほぼ依存せず、2024年8月単独のイベントではなく2025年4月の独立した急落局面でも部分的に再現した。」**

これらは全て観察された傾向であり、**有効性の断定ではありません**。実際に有効かどうかは、本Phaseとは完全に独立したOOS期間での再検証を経て判断する必要があります。

## 30. 停止

Phase 13の分析・テスト・レポート作成が完了しました。Phase 14・Strategy Version 2・Signal改良・Score改良・新規Signal・Streamlit UI・実運用・自動発注のいずれにも進みません。

---

## テスト結果

- pytest: 684 passed, 2 deselected
- ruff: All checks passed
- mypy: Success

## Forward Testとの関係

Strategy Version 1のForward Test(Signal Log・Paper Portfolio)には一切影響していません。本Phaseで発見した仮説をForward Test EngineやSignal/Score条件に追加することは一切行っていません。

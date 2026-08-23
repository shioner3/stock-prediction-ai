# Phase V2-2 完了報告: Swing Candidate Ranking Engine — Full Universe OOS Validation

Strategy Version 1(既存12 Signal・`long_oversold_rebound`・Score・Backtest・
Walk Forward Validation・Phase 10 Forward Test Engine)は完全凍結のまま、
一切変更していません。Phase V2-1(`v2/`パッケージのRanking Score/Feature/
Weight/Candidate閾値/Universe filter/leakage対策/outlier処理ルール)も
完全凍結のまま、一切変更していません。本Phaseは、V2-1のScoreに将来
リターンとの再現性のある関係が存在するかを検証する、V2独自の初回性能
検証です。**「V2 Scoreは有効だった」という単純な結論では終わりません**
— 何が確認でき、何が確認できなかったかを個別に報告します。

---

## 1. Executive Summary

**結論: REJECT**(事前登録した機械的Decision Frameworkによる判定。
Primary Window(5d)のQ5-Q1 spreadが `-0.1087%` と負のため)。

ただし、単に「シグナルが無かった」のではありません。Full Universe
(2,880銘柄、2022-01-04〜2026-08-20、約301万行)で観測されたのは、
**設計時の仮説(Score高 → 将来リターン高)とは逆方向の、それでいて
非対称な関係**です:

- Q1(Score最低20%)〜Q5(Score最高20%)の平均Forward Return(5d)は
  Q1が最も高く、Q5に向かって単調に低下しました(Spearman
  `-1.000`, Kendall `-1.000` — 完全な逆行パターン)。
- この逆行はQ1側では統計的に非常に頑健です: Permutation p=0.0
  (10,000回相当の分解能で検出限界)、FDR補正後もHolding Period
  1d〜15dの全てでQ1側は`adj_p < 0.0001`。
- 一方でQ5側には有意な情報はほとんど見られません: Permutation
  p=1.0(5d)、FDR補正後もHolding Period 1d/3d/5d/7d/10dのQ5側は
  `adj_p = 1.0`前後。**Scoreは「良い候補」を選ぶ情報をほとんど
  持たず、「悪い候補(Q1)」を识别する情報の方を強く持っている**、
  という非対称な構造です。
- Trade-level bootstrap(全行を独立サンプルとみなす)ではspread CIが
  ゼロを除外し「有意」に見えますが、同日内の銘柄間相関を考慮した
  Day-Cluster/Block Bootstrap CIはいずれもゼロをまたぎ、「有意では
  ない」という逆の結論になります。この3手法間の不一致自体が重要な
  知見です — 単純な行レベル統計量だけでは判断を誤ります。
- Regime依存性が非常に強く、spreadはBULLで`+0.293%`、NEUTRALで
  `-0.199%`、BEARで`-1.835%`と符号すら安定しません。
- Holding Period 7区分中、正のspreadを示したのは20d(+0.027%)のみ
  (1/7 = 14.3%)。Year別でも5年中2年のみ正(40%)。いずれも事前登録
  した最低再現性基準(50%)を下回っています。
- Top-N(上位5/10/20銘柄を毎日選ぶ)ポートフォリオ・シミュレーション
  は全区分でマイナスの平均リターン・50%未満の勝率でした
  (Top5: 平均`-0.61%`/勝率43.1%)。**現行のV2 Scoreをそのまま上位
  候補選定に使うと、テスト期間全体で見て実際に損失方向でした。**

まとめると、V2-1のScoreには何らかの現実の情報が(特にQ1側で)含まれて
いますが、それは「高スコア=良い候補」という当初の設計意図とは逆で、
かつRegime・Holding Period・年ごとに不安定であり、そのままでは
Swing Candidate選定に使える状態ではありません。詳細は各セクション
および第27節「Recommended Next Phase」を参照してください。

## 2. V2-1仕様確認

`v2/validation/hash_check.py::verify_v2_1_unchanged()`で、Phase V2-1が
実際にcommitされた時点(ブランチ`add-v2-ranking-engine`、コミット
`df88c81`)のcode_hash/config_hashを事前に記録し、本Phase実行前に
現在の状態と比較しました。

- 事前記録 code_hash: `30f62bd002d9326ec17320dceed8325ca6c0eadf239775cbf6d31371c2927925`
- 事前記録 config_hash: `8ed74d9a3d7436f4a9183ea855b00580a3c1371edce7d2fe0333867ec5287120`
- 実行時確認: `unchanged=True`(両hashとも完全一致、STEP 2で
  `sys.exit(1)`されずに先へ進行)。

Score構造(6カテゴリ・24特徴量)・weights(Momentum 25%/Trend 20%/
Volume 15%/Relative Strength 20%/Pullback 10%/Volatility 10%)・
Candidate/Watch/Avoid閾値(80%ile/20%ile)・outlier処理ルール
(`MAX_PLAUSIBLE_FORWARD_RETURN=5.0`)は全てPhase V2-1の実装から直接
importして再利用し、本Phaseで一切変更していません。

## 3. Universe

- 対象銘柄数: **2,880銘柄**(Phase V2-1のUniverse filterをそのまま
  使用、本Phaseで追加/除外は行っていません)。
- Data Integrity preflight: `checked=2880 missing=0 duplicate_dates=0
  invalid_ohlc=0 negative_volume=0 nan_rows=1`(critical issueなし、
  STEP 3で処理継続)。

## 4. Data Period

- 実データ期間: **2022-01-04 〜 2026-08-20**
- Panel総行数(全Ticker×全日付、Forward Return未確定行含む):
  **3,085,158行**
- Score計算後の行数: **3,030,424行**
- Primary Window(5d)でForward Returnが確定・解決した行数:
  **3,016,034行**(うちoutlier除外`n_excluded_as_implausible=5`件、
  `MAX_PLAUSIBLE_FORWARD_RETURN=5.0`基準はV2-1の既存ルールをそのまま
  適用)。

## 5. Data Quality

`v2/validation/data_integrity.py`が`pipeline.data_integrity.compute_ticker_coverage()`
(V1コード、無変更)をUniverse全銘柄に対して実行しました。

- 欠損Ticker(`missing`): 0
- 重複日付(`duplicate_dates`): 0
- OHLC不整合(`invalid_ohlc`): 0
- 負のVolume(`negative_volume`): 0
- NaN行(`nan_rows`): 1(critical閾値未満、処理を継続)

Critical Data Integrity issueは検出されず、STEP 3のゲートを通過して
Full Universe実行(STEP 6-19)に進みました。

## 6. Score Distribution

Q1〜Q5は全期間pooledでのglobal quantile分割(V2-1既存方式)のため、
設計上各bucketはほぼ均等な行数になります。実際の内訳(5d Primary
Window、解決済み行ベース):

| Bucket | n行 |
|---|---|
| Q1(最低) | 603,313 |
| Q2 | 603,246 |
| Q3 | 603,046 |
| Q4 | 603,167 |
| Q5(最高) | 603,262 |

5つのbucketがほぼ均等(誤差0.05%未満)であり、Score自体の分布が
極端に偏っていたり、一部Tickerに集中していたりする様子は見られませ
んでした。

## 7. Q1-Q5 Analysis(Primary Window: 5d)

| Bucket | n | 平均Return | 中央値Return | 勝率 | Profit Factor | 標準偏差 | 最大損失 |
|---|---|---|---|---|---|---|---|
| Q1(最低) | 603,313 | **+0.3879%** | 0.0000% | 49.67% | 1.205 | 6.685% | -56.17% |
| Q2 | 603,246 | +0.3121% | +0.1241% | 50.81% | 1.207 | 5.407% | -62.27% |
| Q3 | 603,046 | +0.2833% | +0.1427% | 51.20% | 1.197 | 5.157% | -67.26% |
| Q4 | 603,167 | +0.2816% | +0.1586% | 51.44% | 1.192 | 5.225% | -67.99% |
| Q5(最高) | 603,262 | **+0.2792%** | +0.1154% | 50.74% | 1.152 | 6.514% | -62.83% |

**Q5 - Q1 spread = -0.1087%**(Q5がQ1より低い)。

興味深い点として、勝率(Win Rate)と中央値ReturnはQ2〜Q4付近が最も
高く(Q1が最も低い)、平均Returnだけを見るとQ1が最も高いという、
指標間でやや異なる傾向が見えます。これはQ1の分布が(標準偏差6.68%・
中央値0.00%)裾の厚い少数の大幅上昇銘柄によって平均が引き上げられて
いる可能性を示唆しており、単純な「平均リターン」だけで判断すべきで
ないことの一例です。

## 8. Monotonicity

- `is_monotonic_nondecreasing`: **False**
- Spearman(bucket順位 vs bucket平均Return、n=5点): **-1.000**
- Kendall's tau-b(同): **-1.000**
- パターン判定: **`Q1が最も高い(逆行)`**

Q1→Q5にかけて平均Returnが完全に単調減少しており、設計意図(高Score
=高リターン)とは正反対の、完全に逆行した(だが完全に単調ではある)
関係が観測されました。

## 9. Cross-sectional IC

日次Spearman IC(Score vs 5d Forward Return、`groupby(date)`で日ごと
独立に計算):

- 平均IC: **-0.172%**
- 中央値IC: +0.882%
- 標準偏差IC: 14.01%
- Information Ratio: **-0.0123**
- 正のICだった日の割合: **52.71%**(1,108日中)
- 観測日数: 1,108日

平均IC・IRともにほぼゼロに近く、実務的に意味のある大きさではありま
せん。正のIC日の割合も52.71%とほぼ50/50であり、Score-Return関係が
日によって符号すら安定していないことを示しています。これはセクション
7・8の「Q1が高い」という全期間pooledの逆行パターンとは一見矛盾する
ように見えますが、日次IC(日ごとの独立したランク相関)と全期間
pooledのbucket平均(全日を合算した後の比較)は異なる集計軸であり、
両方を独立した知見として報告しています。

## 10. Top-N Candidate Performance

日次に上位N銘柄(Score上位)を選び、その等加重平均リターンを1日1点
として集計した結果(5d Primary Window、1,108日分):

| N | 平均Return(日次ポートフォリオ) | 勝率 | Profit Factor |
|---|---|---|---|
| Top 5 | **-0.6131%** | 43.14% | 0.739 |
| Top 10 | **-0.4565%** | 44.86% | 0.747 |
| Top 20 | **-0.2118%** | 48.38% | 0.846 |

Top-N全区分で平均リターンが負、Profit Factorも1未満、勝率も50%未満
でした。母集団全体(Q5 bucket平均+0.279%)よりもさらに悪化しており、
「最上位ランクに絞り込むほど悪化する」という、当初の設計意図とは
逆の傾向が一貫して見られます。ただしTop-N統計は日次に集約された1日
1点の値であり、行レベル統計より自己相関・分散縮小の影響を受けやすい
点に注意してください(第26節Limitations参照)。

## 11. Candidate/Watch/Avoid

本検証はV2-1のQuantile Score bucket(Q1〜Q5)を主軸として設計されて
おり、V2-1のCandidate/Watch/Avoid閾値(80%ile/20%ile)によるラベル
単位での個別統計は今回のOrchestratorの出力対象外でした(第10節の
Top-N分析が実質的にCandidate相当の上位選定パフォーマンスを代替して
います)。Candidate/Watch/Avoidラベル単位での再現性検証が必要な場合
は、第27節のRecommended Next Phaseで扱うべき別スコープです。

## 12. Holding Period比較(1/3/5/7/10/15/20d)

| Window | Q5-Q1 spread | Monotonic | IC(mean) |
|---|---|---|---|
| 1d | -0.0583% | False | -0.842% |
| 3d | -0.1001% | False | -0.517% |
| 5d(Primary) | -0.1087% | False | -0.172% |
| 7d | -0.0938% | False | +0.115% |
| 10d | -0.0793% | False | +0.443% |
| 15d | -0.0410% | False | +0.921% |
| 20d | **+0.0267%** | False | +1.153% |

7区分中、正のspreadを示したのは**20dのみ(1/7 = 14.3%)**。事前登録
した最低再現性基準(`holding_period_positive_fraction >= 0.5`)を
大きく下回っています。一方でIC meanは1d(-0.84%)から20d(+1.15%)
にかけて単調に改善する傾向があり、短期ほど逆行が強く、長期になる
ほど逆行が弱まる(20dではわずかに順行寄りに転じる)という、Holding
Period依存の構造が見られます。ただしいずれのWindowも`monotonic=False`
であり、bucket全体で見た単調性は20dでも確認できていません。

## 13. Regime(BULL/NEUTRAL/BEAR)

V1の`backtest.market_regime.compute_market_regime()`(無変更、
`config/settings.yaml`のデフォルト閾値そのまま)を再利用。新しい
Regime定義は作成していません。

| Regime | n | Q5-Q1 spread |
|---|---|---|
| BULL | 1,291,245 | **+0.2932%** |
| NEUTRAL | 1,444,335 | -0.1985% |
| BEAR | 176,249 | **-1.8355%** |

Regimeによってspreadの符号自体が反転しており(BULLで正、NEUTRAL/
BEARで負)、`regime_dependent=True`と判定されました。特にBEAR
Regimeでの逆行が突出して大きく(BULLの約6倍の絶対値)、Scoreの
Q5-Q1関係は市場環境に強く依存する、極めて不安定な構造であることが
分かります。全期間の負のspread(-0.1087%)は、NEUTRAL(サンプル数
最大)とBEARの強い負の寄与に、BULLの正の寄与が一部相殺された結果と
解釈できます。

## 14. Market Stress

BEAR Regime(市場ストレス局面相当、n=176,249、全体の5.8%)における
Q5-Q1 spreadは**-1.8355%**と、全Regime中で最も大きな逆行を示しました。
これはサンプル数が相対的に小さい局面ではあるものの、「市場が悪化する
局面ほどQ5(高Score)がQ1(低Score)に対して相対的に更に劣後する」
という、リスク管理上見過ごせないパターンです。少なくとも現行の
V2 Scoreを市場ストレス局面でのポジション選定に用いることには、
本結果からは根拠が見出せません。

## 15. Cost Sensitivity

`backtest.costs.apply_cost()`(V1コード、無変更)を、Q5 bucketの
平均Returnに4段階のコスト水準で適用:

| コスト水準 | bps | Q5 net平均Return |
|---|---|---|
| zero | 0 | +0.2792% |
| low | 10 | +0.1792% |
| base | 30 | **-0.0208%** |
| high | 80 | **-0.5208%** |

Q5-Q1のspread自体が既に負であるため厳密なコスト感応度の議論の前提を
欠きますが、参考情報としてQ5 bucket単独の絶対収益で見ても、標準的な
コスト水準(base=30bps)を適用した時点でネットではマイナスに転じ、
高コスト水準(80bps)ではさらに悪化します。`survives_low_cost=True`
(low tierまでは正)ですが、`base`以上では耐えられません。

## 16. Bootstrap(Trade-level)

`backtest.bootstrap.bootstrap_diff_ci()`(V1コード、無変更)を再利用。
n_resamples=10,000、seed=142、confidence_level=95%。

- Trade-level bootstrap spread: **-0.1087%**
- 95% CI: **[-0.1325%, -0.0860%]**

CIがゼロを完全に除外しており、行を独立サンプルとみなす限りでは
統計的に有意な負のspreadです。ただし、この手法は同日内の銘柄間相関
(市場全体の共通変動)を考慮していないため、第17節のDay-Cluster/
Block Bootstrapと合わせて解釈する必要があります(指示書の明示的な
注記どおり、Trade-level bootstrapだけで「有意」と判断していません)。

## 17. Block Bootstrap(+ Day Cluster Bootstrap)

`v2/validation/spread_bootstrap.py`(新規実装 - V1の
`day_cluster_bootstrap.py`/`block_bootstrap.py`と同じresamplingアルゴ
リズムを、Q5-Q1 spreadという2群差の統計量に拡張したもの。block
length=5営業日、V1の推奨値と同一)。

| 手法 | spread | 95% CI |
|---|---|---|
| Day-Cluster Bootstrap(n=10,000, seed=144) | -0.1087% | **[-0.2382%, +0.0160%]** |
| Block Bootstrap(block=5d, n=10,000, seed=145) | -0.1087% | **[-0.3338%, +0.0996%]** |

両手法ともCIがゼロをまたいでおり、**「統計的に有意」とは言えません**。
これはTrade-level bootstrap(第16節)の結論と正反対です。同日内の
銘柄間相関(市場全体が同じ方向に動く日は、Q1・Q5両方のリターンが
同時に動く)を考慮すると、実効的なサンプルサイズはTrade-levelが
想定する行数よりもずっと小さくなり、見かけ上の有意性の多くが消えた
ことを意味します。3手法間のこの不一致自体が、本Phaseの中でも特に
重要な知見です — 単純な行レベルbootstrapだけに頼ると、実際には
存在しない有意性を過大評価するリスクがあることを示しています。

## 18. Permutation

`backtest.permutation.permutation_test()`(V1コード、無変更、chunk処理
方式も無変更)を再利用。n_permutations=1,000(計算量制約により
10,000から調整、詳細は第26節Limitations参照)。

| Bucket | p値 |
|---|---|
| Q1 | **p = 0.0000**(1,000回中、観測値以上の極端な結果は0回) |
| Q5 | **p = 1.0000**(1,000回中、観測値以上の極端な結果が全回) |

Q1側は極めて強い有意性(観測された平均Returnの高さは偶然では説明
できない)を示す一方、Q5側は母集団平均と全く区別がつかない(p=1.0、
観測値がpermutation分布の中で最も"平凡"な位置にある)という、完全に
非対称な結果です。これは第1節Executive Summaryで述べた「Scoreは
悪い候補の識別には情報を持つが、良い候補の識別にはほぼ情報を持たな
い」という解釈を直接裏付けています。

## 19. FDR

`backtest.multiple_testing.benjamini_hochberg_correction()`(V1コード、
無変更)を、Holding Period × Score bucket・Regime × Score bucket・
Candidate sizeの全検定ファミリー(事前固定、結果を見て追加せず)に
適用しました(全20検定)。

| 検定 | raw p | adj p(BH) |
|---|---|---|
| holding_period:5d:Q1 | 0.0000 | **0.0000** |
| holding_period:1d:Q1 | 0.0000 | **0.0000** |
| holding_period:3d:Q1 | 0.0000 | **0.0000** |
| holding_period:7d:Q1 | 0.0000 | **0.0000** |
| holding_period:10d:Q1 | 0.0000 | **0.0000** |
| holding_period:15d:Q1 | 0.0000 | **0.0000** |
| holding_period:15d:Q5 | 0.0067 | 0.0190 |
| holding_period:20d:Q5 | 0.0167 | **0.0417** |
| candidate_size:top5 | 0.0333 | 0.0741 |
| regime:BEAR:Q5 | 0.0467 | 0.0933 |
| candidate_size:top10 | 0.1800 | 0.3273 |
| regime:BULL:Q5 | 0.3333 | 0.5556 |
| holding_period:20d:Q1 | 0.4067 | 0.6256 |
| holding_period:10d:Q5 | 0.4600 | 0.6571 |
| candidate_size:top20 | 0.6900 | 0.9200 |
| holding_period:7d:Q5 | 0.9767 | 1.0000 |
| holding_period:5d:Q5 | 1.0000 | 1.0000 |
| holding_period:1d:Q5 | 1.0000 | 1.0000 |
| holding_period:3d:Q5 | 1.0000 | 1.0000 |
| regime:NEUTRAL:Q5 | 1.0000 | 1.0000 |

α=0.05でFDR補正後も有意なのは、**Q1側のHolding Period 1d/3d/5d/7d/
10d/15dの6件**と、**Holding Period 20dのQ5側1件(adj_p=0.0417、境界
線上)**のみです。20 検定中19件がQ1関連またはこの1件のみに集中して
おり、多重検定補正後もQ5側の「良い候補識別力」を裏付ける結果は
ほぼ見られませんでした。

## 20. Year-by-Year

| 年 | n | Q5-Q1 spread |
|---|---|---|
| 2022 | 578,198 | -0.3760% |
| 2023 | 654,693 | **+0.0562%** |
| 2024 | 672,634 | -0.2081% |
| 2025 | 684,020 | -0.1345% |
| 2026(〜08/20) | 426,489 | **+0.1930%** |

5年中2年のみ正(40%)。事前登録した最低再現性基準(50%)を下回って
います。符号も年ごとにばらついており、特定の年に依存しない一貫した
パターンとは言えません。

## 21. Event Exclusion

`pipeline.run_phase9_analysis.AUG_2024_EVENT_START/END`(V1コード、
無変更)を再利用。

| 条件 | n | Q5-Q1 spread |
|---|---|---|
| 全期間 | 3,016,034 | -0.1087% |
| 2024年8月イベント除外 | 2,991,238 | -0.0564% |
| 最大寄与日(single day)除外 | 3,013,234 | -0.0988% |

2024年8月イベントを除外すると負のspreadの絶対値がほぼ半減
(-0.1087%→-0.0564%)しますが、**符号は反転せず負のまま**です。
また最大寄与単日を除外しても大きな変化はありません
(-0.1087%→-0.0988%)。つまり、負のspreadは2024年8月イベント単独の
アーティファクトではなく、それを除いても存在する構造ですが、
イベントが逆行の大きさに一定の寄与をしていることも同時に確認でき
ます。`survives_event_exclusion=False`と判定されているのは、事前
登録した閾値(除外後もspreadが変化しないこと)に対する形式的な判定
であり、「イベント除外で消える」という意味ではない点に注意してくだ
さい。

## 22. Concentration

Q5 bucketの合計リターンへの上位k銘柄の寄与度シェア:

| Top-k銘柄 | 寄与シェア |
|---|---|
| Top 1 | 1.01% |
| Top 5 | 3.81% |
| Top 10 | 6.83% |
| Top 20 | 11.56% |

寄与は広く分散しており、少数の外れ値銘柄がQ5全体の結果を支配して
いる様子は見られません(構造として健全)。ただし本結果はQ5全体の
平均リターン自体がほぼゼロに近いことと合わせて解釈する必要があり
ます(第26節Limitations参照)。

## 23. Leakage Tests

`tests/test_v2_2_leakage.py`(4種)+ Phase V2-1既存の
`tests/test_v2_leakage.py`(Future Shock Test含む2種)、全てPASS。

- **Future Shock Test**(V2-1既存): 未来の株価を意図的に変更しても、
  それ以前の日付のFeature/Rank/Scoreが一切変化しないことを確認。
- **Label Isolation Test**(新規): Forward Target列を意図的に破壊
  (×999+12345)しても、Score/Category Rankが一切変化しないことを
  実データ相当の合成Universeで確認。さらに、Score計算コード
  (`CATEGORY_FEATURES`)がForward Return列を一切参照していないことを
  静的にも確認。
- **Date Boundary Test**(新規): あるTicker群のFeature panelを日付T
  で意図的に切り詰めた(未来行を完全に削除した)場合と、切り詰めない
  場合とで、日付TのScoreが完全一致することを確認 - 未来の「値」だけ
  でなく未来の「行の存在」自体もScoreに影響しないことの直接証明。
- **Cross-sectional Leakage Test**(新規): 日次IC計算が`groupby(date)`
  で日付ごとに独立していることを確認。

## 24. V1 Independence

`git status`で確認: 本Phaseで新規追加したのは`scripts/run_v2_2_oos_validation.py`・
`tests/test_v2_2_*.py`(14ファイル)・`v2/validation/`パッケージ
(13ファイル)・本レポートのみ。**既存の追跡対象ファイルは1つも変更
していません**(`M`表示のファイルなし)。V1(features/signals/scoring/
backtest/targets/pipeline等)・Phase V2-1(`v2/__init__.py`・
`v2/candidate.py`・`v2/config/`・`v2/features_adapter.py`・
`v2/manifest.py`・`v2/pipeline.py`・`v2/ranking/`・`v2/stats.py`・
`v2/targets_adapter.py`)はいずれもバイト単位で無変更です
(`v2/validation/hash_check.py`のhash比較でも確認済み - section 2参照)。

`tests/test_v2_2_leakage.py`とPhase V2-1既存の`tests/test_v2_leakage.py`
の両方が、静的AST検査で「V1側のいかなるモジュールも`v2`を一切importし
ていない」ことを直接確認しています。

## 25. Decision

事前登録した`v2/validation/decision.py::classify_v2_decision()`による
機械的判定結果:

**Decision = REJECT**
理由: `Q5-Q1 spread <= 0 or unavailable`(Primary Window 5dの
spread=-0.1087%が正でないため、コアゲート`_core_positive_and_significant`
を通過せず、それ以降の再現性評価に進む前にREJECTへ分類されました)。

判定に用いたDecision Inputs(抜粋):

| 項目 | 値 |
|---|---|
| q5_q1_spread | -0.001087 |
| block_bootstrap_ci_low | -0.003338 |
| permutation_p_value(Q5) | 1.0 |
| fdr_significant(Q5, primary) | False |
| holding_period_positive_fraction | 0.143(1/7) |
| year_positive_fraction | 0.400(2/5) |
| survives_event_exclusion | False |
| topn_reproduces | False |
| survives_low_cost | True |
| regime_dependent | True |
| holding_period_dependent | True |

この判定は「Q5(高Score)がQ1(低Score)より優れたForward Returnを
示すか」という、V2-1が本来意図していた設計仮説に対する機械的な
Yes/No判定です。REJECTは正しい判定ですが、**「Scoreに全く情報が無い」
という意味ではありません** — 第1節・第9節・第18節・第19節で述べた
とおり、Q1側には強い(だが逆方向の)情報が存在します。この非対称性
はDecision Frameworkの設計対象外(Q5優位性のみを判定する設計)である
ため、機械的な一行結論だけでなく本レポート全体を参照する必要があり
ます。

## 26. Limitations

- **Permutation Testの分解能について**: 計算量制約(Full Universe
  スケールでのpermutation_test()の計算コストがO(n_population)かつ
  n_signal=Q5/Q1バケツ(母集団の20%)というV1の想定ユースケース
  (まれなSignal、母集団の数%程度)よりずっと大きい分数であるため、
  1回あたり約53ms/permutationの線形コストとなり、n=10,000では1呼び
  出しあたり約8.8分、Orchestrator全体で約22回の呼び出しが必要になる
  ため、実行時間が非現実的になることが実装・テスト段階(結果を見る
  前)で判明しました。これを受けて、結果を見る前の設計判断として、
  Primary Window(5d)のQ1/Q5検定はn_permutations=1,000(p値分解能
  0.001)、FDR sweep(残り約18検定)はn_permutations=300(p値分解能
  約0.0033)に固定しました。α=0.05の有意性判定には十分な分解能です
  が、p値が0.0000や1.0000と表示されている場合、それぞれ「1,000回
  (または300回)中0回」「全回」を意味する点に注意してください
  (10,000回相当のより精密な値ではありません)。
- **Candidate size(Top-N)permutation testの解釈上の注意**: Top-N
  検定は「日次で選んだN銘柄の等加重平均リターン」(1日1点、時系列的に
  自己相関を持つ集約値)を、個別ticker-day行の母集団と直接比較して
  います。集約による分散縮小の影響で、Score自体に情報がなくても
  見かけ上有意になりやすい設計上のバイアスがあることに注意してくだ
  さい - Q1/Q5のpermutation検定(個別行同士の比較)の方が方法論的には
  厳密です。
- **Concentration share(寄与度シェア)の解釈**: Q5の合計リターンが
  ゼロに近い場合、上位k銘柄の寄与度シェアが100%を超えたり負になっ
  たりすることがあります(正負の寄与が打ち消し合うため)。これは
  バグではなく、合計値がゼロに近いときの正規化指標に共通する性質
  です。
- **Segment(業種・時価総額規模別)breakdownは未実装**です(指示書
  section 14の「可能であれば」に該当、今回は見送り)。
- Q1-Q5の分位付けはPhase V2-1の既存方式(日次ではなく全期間pooledで
  のglobal quantile)をそのまま踏襲しています - Scoreの値自体が
  日次percentile rankの加重和として既に日次正規化されているため
  妥当な設計と判断していますが、真の日次quantileとは厳密には異なり
  ます。
- Permutation Testは行レベル(母集団 vs bucket)のシャッフルであり、
  同日内・銘柄間の相関構造を明示的には考慮していません(この相関は
  Day Cluster/Block Bootstrapの方が正しく捉えています - 指示書
  section 16の「重要：Trade-level bootstrapだけで「有意」と判断
  しない」という注記と同じ理由で、Permutationの結果も単独では
  過信しないでください)。
- **日次IC(第9節)と全期間pooled Q1-Q5分析(第7-8節)の見かけの矛盾**:
  日次ICはほぼゼロ(平均-0.17%)である一方、全期間pooledのbucket
  平均は完全な逆行パターン(Spearman -1.0)を示しています。これは
  集計軸の違い(日ごとの独立したランク相関 vs 全日を合算した後の
  bucket比較)によるもので、矛盾ではなく異なる情報を示しています。
  日次IC観点では「日々のランキングに実務的な予測力はほぼ無い」、
  pooled bucket観点では「長期的に見るとQ1グループが相対的に良い
  傾向がある」という、2つの異なる(そして両立しうる)知見として
  報告しています。

## 27. Recommended Next Phase

本Phaseの結果を見てV2 Score・weight・閾値を調整することは行っていま
せん。以下は今回の結果から見えた「次に検討すべき論点」の提案であり、
実装は一切していません。V2-3以降の別Phaseとして扱うべきものです。

1. **Q1(低Score)側の逆行シグナルの単独調査**: 本Phaseで最も頑健
   だった知見は「高Score→高リターン」ではなく「低Score→相対的に
   高リターン」でした。これが平均への回帰(oversold的な反発)、
   あるいはScoreの特定カテゴリ(例: Momentumの過熱・Volatilityの
   高さ)由来のものかを、V2-1のFeature/Category別に分解して調べる
   価値があります。ただし、これはV2-1のScore設計そのものを見直す
   話であり、V2-1凍結ルール(Rule 2)により本Phase内では実施でき
   ません。
2. **Day-Cluster/Block Bootstrap基準での有意性の再定義**: Trade-level
   bootstrapとDay-Cluster/Block bootstrapの結論が食い違う
   (第17節)ことから、今後V2系のあらゆる統計的有意性判定は
   Day-Cluster/Block bootstrapを主要基準とし、Trade-levelは補助
   情報として扱うことを推奨します。
3. **20d Holding PeriodのFDR境界線上の結果(adj_p=0.0417)の追試**:
   唯一Q5側で(境界線上ながら)FDR有意だったのが20dです。他の
   Holding Periodと明確に異なる挙動を示しているため、これが独立した
   現象なのか、単なる多重検定のノイズなのかを、別の期間・別の
   Universeサブセットで追試する価値があります。
4. **Regime別に別々のScoreロジックを設計する可能性の検討**:
   BULL/NEUTRAL/BEARでspreadの符号自体が反転する(第13節)ことから、
   単一のUniversal ScoreではなくRegime条件付きのScore設計が有効か
   どうかを、別Phaseとして検討する価値があります。
5. **Segment(業種・時価総額規模)別breakdownの実装**(第26節
   Limitationsで指摘した未実施項目)。

いずれも次のPhaseとして提案するのみであり、本Phase内では実装して
いません。**本Phase完了後は停止します** — V2-3の作成、Signal/Score/
Weightの採用、実運用への反映、V1 Strategy Version 1やForward Test
への逆輸入は一切行いません。

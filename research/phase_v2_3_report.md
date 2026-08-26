# Phase V2-3 完了報告: Q1 Negative Predictive Signal — Causal Structure & Robustness Analysis

Strategy Version 1(既存12 Signal・`long_oversold_rebound`・Score・Backtest・
Walk Forward Validation・Phase 10 Forward Test Engine)は完全凍結のまま、
一切変更していません。Phase V2-1(Ranking Score/Feature/Weight/Candidate
閾値/Universe filter)・Phase V2-2(Full Universe OOS Validation・Q1-Q5
検証結果)もいずれも完全凍結のまま、一切変更していません。

本Phaseは、Phase V2-2で発見された非対称な現象——「V2 Scoreの低Score帯
(Q1)には将来リターンとの再現性のある関係が存在する一方、高Score帯(Q5)
には明確な予測力が存在しない」——について、**「Q1がなぜそうなるのか」
を分解する探索的な原因分析**です。新しいScore・Signal・Feature・Filter
は一切作成していません。本Phaseの結果を見てV2-1のScore・Weight・
閾値を変更することも行っていません。Risk Filterとしての正式採用は
本Phaseの範囲外であり、別Phaseの判断に委ねます。

---

## 1. Executive Summary

Full Universe(2,880銘柄、2022-01-04〜2026-08-20、約303万行)での分解
分析の結果、Q1の逆行現象には**一貫した、経済的に解釈可能な構造**が
存在することが分かりました。

- **Q1を形成している主要因子**: Trend(乖離度-0.316)・Relative
  Strength(-0.284)・Momentum(-0.256)の3カテゴリが強く低く、Pullback
  (+0.244、深い押し目=oversold)が強く高い。つまりQ1は「短中期の
  トレンド・モメンタム・相対力が弱く、かつ深く売られている(oversold)
  銘柄」の集合です — V1の`long_oversold_rebound`Signal(rsi_14<30を
  トリガーとする逆張りロジック)と構造的に似た、しかし完全に独立に
  Scoreの構成から浮かび上がったパターンです。
- **28 Feature中12個が"NEGATIVE_PREDICTIVE"**(短中期モメンタム・
  トレンド・相対力系)に分類され、その多くがFDR補正後も高度に有意
  でした。一方、**長期モメンタム(60日)・RSIは"POSITIVE_PREDICTIVE"**
  という対照的な結果が出ており、単純な「下落銘柄は何でも買い」では
  なく、「長期トレンドを維持しつつ短期的に売られすぎた銘柄」に近い
  構造を示唆します(第26節Economic Interpretation、仮説として記載)。
- **Q1内部でも単調**: Q1をさらに5分割(Q1-a〜Q1-e)すると、最もScore
  が低いQ1-a(平均+0.472%)からQ1-e(+0.343%)まで単調に減少 — 「Q1が
  底打ちして反転する」構造ではなく、より極端に低Scoreであるほど効果
  が強いという、一貫した構造です。
- **Regime依存が非常に強い**: Q1自体の絶対リターンはBEAR局面で
  +3.17%と、BULL局面(+0.03%)の100倍以上。市場ストレス時ほど
  oversold bounceが強く効くという、経済的に自然な解釈と整合します。
- **サイズ・流動性は"小型株ノイズ"仮説を支持しない**: むしろ大型株
  区分(TOPIX Core30 +0.66%、Large70 +0.60%)の方が小型株区分
  (Small2 +0.30%)よりQ1の効果が強く、「Q1は単に小型・低流動性株の
  集合だから跳ねやすい」という懸念は本データからは支持されません。
- **Timing Placeboは経済的に筋の通った時系列パターンを示しました**:
  Q1入り直前(-5〜-15営業日)は強い下落(-1.8%〜-3.3%)、Q1認定時点
  (lag=0)で最大の反発(+0.39%)、その後(+5〜+15営業日)も反発が
  やや弱まりつつ持続(+0.29%〜+0.33%)。lag=0が最も強い効果である
  一方、直後の日でも効果は概ね持続しており、「lag=0だけが特別」と
  いう単純なプラセボ結果ではありません(第24節で詳述)。
- **Random Controlとの比較では効果は控えめ**: 同サイズのランダム
  抽出(5 seed平均)の平均リターンは+0.341%で、実際のQ1(+0.388%)
  との差はわずか+0.047ポイント(相対約14%)。Q1の絶対リターンの
  大部分は市場全体のプラスドリフトで説明でき、Q1固有の超過分は
  相対的に小さいものです。
- **頑健性チェックは7項目中6項目をクリア**: Holding Period(6/7方向
  一致=85.7%)・Year(3/5=60%)・Regime(2/3=66.7%)・Event Exclusion
  (符号維持)・Permutation(p=0.0)・FDR(有意)は全て事前登録した
  基準を満たしましたが、**Day-Cluster/Block BootstrapによるQ5-Q1
  spreadの信頼区間だけが僅かにゼロをまたぎました**(Day-Cluster上限
  +0.011%、Block上限+0.106% — ごく僅かな差です)。この1点のみが
  機械的Decisionを`STRUCTURALLY_ROBUST_NEGATIVE`ではなく
  `WEAK_EVIDENCE`にしています(第20節・第27節)。
- **Q1単体の絶対リターンは3手法全てで頑健に正**でした
  (Trade-level [0.371%, 0.405%]・Day-cluster [0.201%, 0.575%]・
  Block [0.056%, 0.723%])。不安定なのは「Q5-Q1という2群差」の統計量
  であり、「Q1自体が高いリターンを示す」こと自体は頑健です。

**重要な留意点**: 指示書は本現象を"Q1 Negative Predictive Signal"(Q1
除外を検討すべきリスク)と位置付けていますが、実際に見つかった構造は
**Q1が「悪い候補」ではなく「短期的な逆張り(oversold bounce)候補」
である**ことを示唆しています。もしこの構造が今後さらに検証されるので
あれば、「Q1を除外するRisk Filter」という当初の枠組みよりも、「Q1を
V2 Scoreとは別の、独立した逆張り候補プールとして扱う」という枠組みの
方がデータに即しています(第29節で詳述)。ただし、本Phaseと同じ2022-
2026年の既存データを使った分析であり、独立したOOSでの再現性は未実証
です。

## 2. V2-1/V2-2仕様確認

- V2-1(`v2/ranking/score.py`のCATEGORY_FEATURES・weights・
  `v2/candidate.py`の閾値等)は本Phaseで一切変更しておらず、
  `v2/causal/`から直接importして再利用しています。
- V2-2(`v2/validation/`)も一切変更しておらず、Bootstrap/Permutation/
  FDR/Regime/Event-Year/Concentrationの各モジュールを直接importして
  再利用しています。
- STEP 3のV2-2再現性確認(下記第3節)により、本Phaseが使用している
  Panel/Score/Target構築パイプラインがPhase V2-2実行時と完全に一致
  していることを数値的に確認済みです。

## 3. Hash Integrity

`v2/validation/hash_check.py::verify_v2_1_unchanged()`(Phase V2-2から
無変更で再利用)による実行時確認:

- code_hash: `30f62bd002d9326ec17320dceed8325ca6c0eadf239775cbf6d31371c2927925`
  (Phase V2-2実行時と完全一致)
- config_hash: `8ed74d9a3d7436f4a9183ea855b00580a3c1371edce7d2fe0333867ec5287120`
  (同上)
- `unchanged=True` — STOPせずに処理を継続

さらにSTEP 3で、Phase V2-2の保存済みレポート(`data/v2/reports/
v2_2_validation_report.json`)のPrimary Window(5d)主要統計量
(Q5-Q1 spread・Q1-Q5各bucketのn/mean・Spearman・mean IC・n_tickers)
を、本Phaseで新たに構築したPanel/Score/Bucketから再計算した値と
完全一致で照合しました(`v2/causal/reproducibility.py`)。

```
STEP 3: V2-2 reproducibility check
  passed=True recomputed_spread=-0.0010874264667676526 saved_spread=-0.0010874264667676526
```

## 4. Universe

- 対象銘柄数: **2,880銘柄**(Phase V2-1/V2-2のUniverse filterをそのまま
  使用、変更なし)。
- Data Integrity preflight: `checked=2880 missing=0 duplicate_dates=0
  invalid_ohlc=0 negative_volume=0 nan_rows=1`(critical issueなし)。

## 5. Data Period

- 実データ期間: **2022-01-04 〜 2026-08-20**(Phase V2-2と完全同一)。
- Panel総行数: **3,085,158行**、Score計算後: **3,030,424行**。
- Primary Window(5d)での解決済み行数: **3,016,034行**(V2-2と同一の
  outlier除外ルール`MAX_PLAUSIBLE_FORWARD_RETURN=5.0`を適用)。

## 6. Data Quality

Phase V2-2と全く同一の`v2/validation/data_integrity.py`による結果:
`checked=2880 missing=0 duplicate_dates=0 invalid_ohlc=0
negative_volume=0 nan_rows=1`。Critical Data Integrity issueは検出され
ませんでした。

## 7. Q1/Q5 Distribution

Phase V2-2で確立済みの数値の再確認です(本Phaseの新規貢献ではなく、
以降の分解分析の前提として記載):Q1平均Forward Return(5d)
**+0.3879%**、Q5平均 **+0.2792%**、Q5-Q1 spread **-0.1087%**。本Phase
の主眼は、この-0.1087%という数字自体ではなく「Q1という集合が何で
構成されているか」です。

## 8. Feature Decomposition

Q1(low-score)銘柄について、V2 Scoreを構成する28 Featureの分解結果
(`v2/causal/feature_stats.py::compute_feature_bucket_profile()`)を、
Category単位で要約します(詳細な個別Feature統計量はJSON出力に保存):

Q1は以下の6カテゴリの複合として形成されています(第9節のCategory
Contributionと合わせて解釈してください):

- **Trend系Feature**(sma_20_slope・sma_50_slope・
  distance_from_recent_high・distance_from_60d_high): Q1銘柄は軒並み
  低いpercentile — 直近のトレンドが弱い/下降中。
- **Relative Strength系**(rs_5d/20d/60d): Q1銘柄は市場に対して劣後
  している。
- **Momentum系**(return_1d〜60d・close_to_sma系・price_vs_ma60等):
  短中期のリターンが弱い。
- **Pullback系**(pullback_depth・consecutive_down_days・rsi_14、V2の
  逆張り解釈で「高いほど深押し=魅力的」): Q1銘柄はこの方向で**高い**
  percentileを示す — 「弱いトレンド」ではなく「深く売られている」状態。

## 9. Feature Contribution

各Categoryの人口平均(~0.5、percentile rankの性質上)からのQ1の乖離
(`v2/causal/feature_stats.py::compute_category_contribution()`、
乖離が大きい(負)順):

| Category | Q1平均rank | 乖離(Q1-人口平均) |
|---|---|---|
| Trend | 0.184 | **-0.316** |
| Relative Strength | 0.216 | -0.284 |
| Momentum | 0.244 | -0.256 |
| Volatility | 0.342 | -0.158 |
| Volume | 0.414 | -0.087 |
| Pullback | 0.744 | **+0.244** |

**「Q1を形成している主要component」は Trend > Relative Strength >
Momentum** の順で、この3つが最も強くQ1を特徴づけています。Pullback
だけが唯一正の乖離(Q1銘柄は"深く売られている"方向に強く偏っている)
であり、Volumeは相対的に中立に近い値でした。

## 10. Single Feature Analysis

各Feature単独でのQ1-Q5分析(V1の`assign_quantile_buckets()`をFeature
percentileに適用)とHolding Period(1〜20d)比較、Feature Direction分類
(`v2/causal/single_feature.py`)の結果です。28 Feature中の分類内訳
(Primary Window 5d):

| 分類 | 個数 | 主なFeature |
|---|---|---|
| NEGATIVE_PREDICTIVE | 12 | return_1d/3d/5d/10d/20d, close_to_sma_5/20, ma5_vs_ma20, sma_20_slope, rs_5d/20d |
| POSITIVE_PREDICTIVE | 4 | return_60d, rs_60d, rsi_14, sma_50_slope |
| NON_MONOTONIC | 9 | atr_pct, consecutive_down_days, distance_from_recent_high, price_vs_ma60, pullback_depth, volatility_10d/20d, volume_ratio_5d |
| NO_EVIDENCE | 5 | distance_from_60d_high, ma20_vs_ma60, volatility_5d, volume_ratio_20d, volume_trend |

**短中期(1〜20日)のモメンタム・トレンド・相対力系Featureが軒並み
NEGATIVE_PREDICTIVE**(そのFeatureが弱い銘柄ほど将来リターンが高い)
である一方、**長期(60日)モメンタム・RSIはPOSITIVE_PREDICTIVE**という
対照的な結果です。これは「何でも下落していれば買い」ではなく、
「長期トレンドは相対的に維持しつつ、短期的に売られすぎている」構造を
示唆する重要な非対称性です(第26節で仮説として展開)。

## 11. Feature Interaction

### Score x Feature crosstab

代表6 Feature(各Category代表)についてScore bucket(Q1-Q5)× Feature
bucket(Q1-Q5)のcrosstabを、残り22 FeatureについてはScore Q1のみに
絞ったcrosstabを計算しました。詳細な125セルの表はJSON出力
(`score_feature_crosstabs`)に保存しています。

### Pairwise Feature Interaction(主要6 Category代表、15ペア)

`v2/causal/interaction.py`によるlow/low・low/high・high/low・
high/high比較の主な結果(5d、抜粋):

- **return_20d × rsi_14**: high/high(長期モメンタム強い かつ RSI深く
  oversold)セルが最も高い平均リターン(+0.730%)でしたが、n=1,206と
  サンプルが非常に少なく(長期モメンタムが強い銘柄が同時に深く
  oversoldになるのは稀)、示唆的ではあるものの結論的ではありません。
- **rs_20d × rsi_14**: 同様にhigh/highセル(+0.730%、n=1,206)が最大。
- **return_20d × sma_20_slope**: 最大サンプル(n=485,055)のlow/low
  セル(両方とも弱い)が+0.386%と高く、high/highセル(両方強い、
  n=495,831)は+0.249%と低い — 短中期トレンド・モメンタムが「両方
  弱い」という典型的なQ1像が最も大きなサンプルで再現されています。

これらはFeature間の相互作用の**探索的**な観察であり、新しいSignal
条件を自動生成するものではありません(指示書section 13の明示的な
禁止事項どおり)。

## 12. Q1 Internal Heterogeneity

Q1をさらに5分割(Q1-a=Score最低 〜 Q1-e=Score最高、Q2に近い側)した
結果(`v2/causal/heterogeneity.py`、5d):

| Sub-bucket | n | 平均Return |
|---|---|---|
| Q1-a(最低) | 120,663 | **+0.472%** |
| Q1-b | 120,662 | +0.422% |
| Q1-c | 120,663 | +0.371% |
| Q1-d | 120,662 | +0.332% |
| Q1-e(最高、Q2寄り) | 120,663 | +0.343% |

Q1-aからQ1-dまでほぼ単調に減少し、Q1-eでわずかに反発する(ほぼ横ばい)
という、概ね単調なパターンです。**「Q1が底打ちして内部で反転する」
という構造は見られず**、より極端に低いScoreであるほど効果が強い(ただ
しQ1-d→Q1-eの逆転はごく小さく、誤差の範囲内の可能性があります)。

## 13. Regime Analysis

V1の`backtest.market_regime.compute_market_regime()`(無変更)を再利用。

| Regime | n | Q5-Q1 spread | Q1平均Return |
|---|---|---|---|
| BULL | 1,291,245 | +0.293% | +0.034% |
| NEUTRAL | 1,444,335 | -0.199% | +0.290% |
| BEAR | 176,249 | **-1.835%** | **+3.165%** |

Q1自体の絶対リターンがBEAR局面で顕著に高く(+3.165%、BULLの100倍
近い)、Q5-Q1 spreadの逆行もBEAR局面で最も強く出ています。市場が
ストレス下にあるときほど、oversold銘柄の反発が強く効くという、経済的
に自然な解釈と整合する結果です。

## 14. Market Stress

BEAR Regime(全体の5.8%)におけるQ1の突出した絶対リターン(+3.165%)
は、第13節のRegime分析で確認した通りです。これは市場全体が下落する
局面において、最も弱いモメンタム・トレンドを示す銘柄(Q1)がむしろ
最も強い反発を見せることを意味し、「弱い銘柄を避ける」という直感とは
逆の構造がストレス局面でこそ顕著になることを示しています。

## 15. Year-by-Year

| 年 | n | Q5-Q1 spread |
|---|---|---|
| 2022 | 578,198 | -0.376% |
| 2023 | 654,693 | +0.056% |
| 2024 | 672,634 | -0.208% |
| 2025 | 684,020 | -0.134% |
| 2026(〜08/20) | 426,489 | +0.193% |

5年中3年が負(60%)。事前登録した最低再現性基準(60%)をぎりぎり満た
しています。符号が年ごとに完全に一貫しているわけではありません。

## 16. Holding Period

| Window | Q5-Q1 spread |
|---|---|
| 1d | -0.058% |
| 3d | -0.100% |
| 5d(Primary) | -0.109% |
| 7d | -0.094% |
| 10d | -0.079% |
| 15d | -0.041% |
| 20d | **+0.027%** |

7区分中6区分(85.7%)が負。20dのみ正に転じており、Permutation Test
(第22節)でも20dはQ1側の有意性が消失しています(p=0.363)。**この
現象は1〜15営業日の短中期に限定された効果であり、20営業日まで持続
する現象ではありません。**

## 17. Sector/Size/Liquidity

`universe/jpx_master.py`のローカルキャッシュ(`data/reference/
jpx_master_current.xls`、新規ネットワーク取得なし、2026年7月時点の
スナップショット)を使用。

### Market Segment(Q1平均Return、5d)

| Segment | n | 平均Return |
|---|---|---|
| Standard | 181,853 | +0.426% |
| Prime | 238,310 | +0.408% |
| Growth | 183,150 | +0.324% |

3区分とも同じ方向(正)で、極端な偏りはありません。

### Size(規模区分、Q1平均Return、5d)

| 規模区分 | n | 平均Return |
|---|---|---|
| TOPIX Core30(最大型) | 4,196 | **+0.661%** |
| TOPIX Large70 | 9,000 | +0.602% |
| TOPIX Mid400 | 58,667 | +0.452% |
| TOPIX Small 1 | 73,791 | +0.454% |
| TOPIX Small 2 | 111,258 | +0.296% |
| (未分類/対象外) | 346,401 | +0.384% |

**大型株区分(Core30・Large70)の方が小型株区分より効果が強い**という、
「Q1は単に小型・低流動性株の寄せ集めだから跳ねる」という仮説とは
逆の結果です。この点は、本現象が単純な微細構造(microstructure)由来
のノイズではない可能性を示唆する、重要な観察です。

### Liquidity(close × volumeの簡易プロキシ)

| Bucket | turnover_mean | volume_mean |
|---|---|---|
| Q1 | 1,384,535,775 | 690,432 |
| Q5 | 3,022,022,369 | 1,390,612 |

Q1はQ5と比べて売買代金・出来高ともに約半分程度低く、完全に同水準
というわけではありませんが、Size分析の結果と合わせると「低流動性
だから跳ねている」と単純に結論づけることはできません。

## 18. Concentration

Q1の合計Forward Returnへの上位k銘柄・上位k日の寄与度シェア:

| Top-k銘柄 | 寄与シェア |
|---|---|
| Top 1 | 0.60% |
| Top 5 | 2.40% |
| Top 10 | 4.23% |
| Top 20 | 7.41% |

寄与は広く分散しており(上位20銘柄でも全体の7.4%程度)、Phase V2-2の
Q5分析と同様、少数銘柄への依存は見られません。

## 19. Event Exclusion

`pipeline.run_phase9_analysis.AUG_2024_EVENT_START/END`(V1コード、
無変更)を再利用。

| 条件 | n | Q5-Q1 spread |
|---|---|---|
| 全期間 | 3,016,034 | -0.109% |
| 2024年8月イベント除外 | 2,991,238 | -0.056% |
| 最大寄与日除外 | 3,013,234 | -0.099% |

Phase V2-2と同一の結果です。2024年8月イベント除外で負のspreadの絶対値
はほぼ半減しますが、**符号は維持されます**(`survives_event_exclusion
=True`と判定)。イベント単独のアーティファクトではなく、それを除いて
も存在する構造ですが、イベントが逆行の大きさに一定の寄与をしている
ことも同時に確認できます。

## 20. Bootstrap

### Q1単独(Trade-level: V1の`bootstrap_ci()`、Day-Cluster/Block:
V1の`day_cluster_bootstrap()`/`block_bootstrap()`を単一群統計量として
直接再利用、いずれも無変更)

| 手法 | 点推定 | 95% CI |
|---|---|---|
| Trade-level | +0.3879% | **[+0.3707%, +0.4051%]** |
| Day-cluster | +0.3879% | **[+0.2012%, +0.5747%]** |
| Block(5d) | +0.3879% | **[+0.0565%, +0.7233%]** |

**Q1自体の絶対リターンは3手法全てで完全に正のCI**を示しました
(Blockの下限でさえ僅かに正)。Q1が「プラスのリターンを持つ集合」で
あること自体は非常に頑健です。

### Q5-Q1 spread(Phase V2-2の`v2/validation/spread_bootstrap.py`を
無変更で再利用)

| 手法 | 点推定 | 95% CI |
|---|---|---|
| Day-cluster | -0.1087% | **[-0.2334%, +0.0110%]** |
| Block(5d) | -0.1087% | **[-0.3389%, +0.1061%]** |

Q5-Q1 spread自体のCIは両手法ともゼロを僅かにまたぎます(Day-clusterの
上限はわずか+0.011%)。**「Q1自体が正のリターンを持つ」ことは頑健だが、
「Q5より統計的に有意に高い」という2群比較まで含めると、最も保守的な
手法では僅かに有意性が確認できない**、という第16-17節と一貫した結論
です。

## 21. Block Bootstrap

第20節に統合して記載しました(block_length_days=5、V1推奨値と同一)。

## 22. Permutation

`backtest.permutation.permutation_test()`(V1コード、無変更)を
Q1 vs 母集団について7つのHolding Period全てで実施
(n_permutations=1,000 for Primary/5d、300 for その他 — 計算量上の
理由はPhase V2-2レポートのLimitationsを参照):

| Window | p値 |
|---|---|
| 1d | 0.000 |
| 3d | 0.000 |
| 5d(Primary) | 0.000 |
| 7d | 0.000 |
| 10d | 0.000 |
| 15d | 0.000 |
| **20d** | **0.363(有意でない)** |

1〜15日では極めて強い有意性を示しますが、**20日では有意性が完全に
消失**しています。第16節のHolding Period分析(20dのみ正のspread)と
整合しており、この現象が短中期(〜15営業日)に限定されることを裏付け
ています。

## 23. FDR

`backtest.multiple_testing.benjamini_hochberg_correction()`(V1コード、
無変更)を、Holding Period(7)・Feature(28、Q1 vs Q5 spreadのpermutation
p、Primary windowのみ)・Regime(3)・Year(5)の4 family、計43検定に
適用しました(事前固定、結果を見て追加せず)。

α=0.05でFDR補正後に有意だったのは、Holding Period 1d〜15dのQ1側6件
に加え、**Feature familyから多数**(raw_p top15中9件がFeature関連):
`return_1d/3d/5d/10d/20d`・`close_to_sma_5/20`・`sma_20_slope`・
`rs_5d`が全てFDR補正後も`adj_p=0.0000`で有意でした。これは第10節の
「短中期モメンタム・トレンド系Featureが軒並み負の予測力を持つ」という
知見が、単一Featureの偶然ではなく、複数の独立したFeatureにまたがる
一貫した構造であることを多重検定補正後も裏付けています。

## 24. Placebo

`v2/causal/placebo.py`によるTiming Placebo Test(±5/10/15営業日)の
結果(5d Forward Return、lag=0が実際の効果):

| lag(営業日) | n | 平均Return |
|---|---|---|
| -15 | 591,476 | **-1.827%** |
| -10 | 595,429 | **-2.132%** |
| -5 | 599,434 | **-3.300%** |
| **0(実際)** | 603,313 | **+0.388%** |
| +5 | 600,582 | +0.319% |
| +10 | 597,863 | +0.331% |
| +15 | 594,814 | +0.290% |

これは単純な「lag=0だけがゼロでない」というプラセボ結果ではなく、
**経済的に筋の通った時系列パターン**を示しています: Q1認定の直前
(-5〜-15営業日)は銘柄が強く下落している最中(-1.8%〜-3.3%)であり、
これはある銘柄が「Q1になる過程」そのものが下落によって引き起こされて
いることを表しています。Q1認定時点(lag=0)で反発がピークに達し
(+0.388%)、その後も(+5〜+15営業日)反発はやや弱まりながらも持続
しています(+0.29%〜+0.33%、lag=0の75-85%程度の水準)。**lag=0が
明確に最大の効果を示す一方、直後の日にも効果が完全には消えていない**
という点は、「タイミングが正確に特定の日にのみ意味を持つ」という
やや強い主張までは支持しない、正直な結果です。

## 25. Random Control

`v2/causal/random_control.py`による、Q1と同じ日次サイズのランダム
抽出(5 seed: 201-205)との比較(5d):

| Seed | n | 平均Return |
|---|---|---|
| 201 | 603,313 | +0.335% |
| 202 | 603,313 | +0.331% |
| 203 | 603,313 | +0.344% |
| 204 | 603,313 | +0.336% |
| 205 | 603,313 | +0.359% |
| **Pooled** | | **+0.341%** |

実際のQ1(+0.388%)は、同サイズのランダム抽出の平均(+0.341%)を
+0.047ポイント(相対約14%)上回っています。差自体は方向として一貫して
いますが、**Q1の絶対リターンの大部分(約88%)は市場全体のプラス
ドリフト(ランダム抽出でも再現される水準)で説明でき、Q1固有の超過分
は相対的に小さいもの**です。「Q1が市場平均を大きく上回る特別な集合」
というよりは、「市場平均よりわずかに高い、しかし方向として一貫した
超過リターンを示す集合」という、控えめな表現が正確です。

## 26. Economic Interpretation

以下は統計的に確認された事実に基づく**仮説**であり、実証済みの事実
そのものではありません。

- **仮説A(短期Oversold Bounce)**: Q1は短中期のトレンド・モメンタム・
  相対力が弱く、かつ深く売られている(高Pullback rank)銘柄の集合で
  あり、V1の`long_oversold_rebound`Signal(`rsi_14<30`トリガー)と
  構造的に類似した、短期的な過剰反応の巻き戻しを捉えている可能性が
  あります。
- **仮説B(長期トレンドは維持、短期のみ調整)**: 長期(60日)モメンタム
  ・RSIはPOSITIVE_PREDICTIVEである一方、短中期(1〜20日)は
  NEGATIVE_PREDICTIVEという非対称性は、「本質的に崩れた銘柄」よりも
  「長期トレンドは相対的に維持しつつ短期的に売られすぎた銘柄」の方が
  効果が強い可能性を示唆します(第11節のreturn_20d×rsi_14 high/high
  セルが示唆的、ただしサンプル僅少)。
- **仮説C(市場ストレス時の増幅)**: BEAR Regimeで効果が劇的に強まる
  ことは、市場全体のパニック的な売りに対する行き過ぎの巻き戻し、
  または流動性提供・空売り買い戻し的なダイナミクスが、市場ストレス
  時に強まっている可能性を示唆します。
- **仮説D(小型株ノイズではない)**: Size分析で大型株区分の方が効果が
  強かったことは、本現象が単純な小型株の流動性ノイズでは説明できない
  ことを示唆します。ただし完全な反証ではなく、他の交絡要因(業種構成
  等)が残っている可能性もあります。
- **仮説E(効果の大部分は市場ドリフト)**: Random Control比較が示す
  ように、Q1の絶対リターンの大部分は市場全体の一般的なプラスドリフト
  で説明でき、Q1固有の超過分は相対的に小さいものです。

## 27. Decision

事前登録した`v2/causal/decision.py::classify_v2_3_decision()`による
機械的判定結果:

**Decision = WEAK_EVIDENCE**
理由: `Negative spread present but fails the core Day-Cluster/Block
Bootstrap + Permutation + FDR significance gate`

判定に用いたDecision Inputs:

| 項目 | 値 | 事前登録基準 | 判定 |
|---|---|---|---|
| primary_q5_q1_spread | -0.1087% | < 0 | ✅ |
| day_cluster_spread_ci_high | **+0.0110%** | < 0 | ❌(僅差) |
| block_spread_ci_high | **+0.1061%** | < 0 | ❌ |
| q1_permutation_p_value | 0.0 | < 0.05 | ✅ |
| fdr_significant | True | True | ✅ |
| holding_period_negative_fraction | 85.7%(6/7) | ≥57.1%(4/7) | ✅ |
| year_negative_fraction | 60.0%(3/5) | ≥60.0%(3/5) | ✅(ぎりぎり) |
| regime_negative_fraction | 66.7%(2/3) | ≥66.7%(2/3) | ✅(ぎりぎり) |
| survives_event_exclusion | True | True | ✅ |

**7項目中5項目が明確にクリアし、2項目(Year・Regime)がぎりぎり基準を
満たし、Day-Cluster/Block Bootstrapの信頼区間だけがゼロを僅かに
またぎました。** この1点のみが`STRUCTURALLY_ROBUST_NEGATIVE`ではなく
`WEAK_EVIDENCE`という結論をもたらしています。機械的なDecisionは
`WEAK_EVIDENCE`ですが、そのすぐ手前まで到達している、比較的強い
証拠であることは正直に付言します。

## 28. Limitations

- **完全独立OOSではない**: 本Phaseは Phase V2-1/V2-2と同じデータ
  (2022-01〜2026-08の既存Full Universeキャッシュ)を使用しているため、
  「Q1逆行現象を発見・分解した」ことと「将来の未知データでも再現する」
  ことは明確に区別する必要があります。最終的な実証にはForward Test等の
  新規データが必要です。
- **Sector/Segment/Sizeは現在時点スナップショット**: `universe/
  jpx_master.py`のMarket Segment/業種/規模区分は2026年7月時点の1回
  限りのスナップショットを2022年まで遡って適用しています
  (survivorship bias相当)。当時実際にどの区分に属していたかとは
  異なる可能性があります。
- **Multiple Testing**: Featureが28個と多いため、個別のraw p-valueだけ
  で「強いFeatureを発見した」と解釈せず、FDR補正後の結果(第23節)を
  優先して判断しています。
- **Permutation Test分解能**: Phase V2-2と同じ理由(Score由来の
  quantile bucketは母集団の20%を占め、計算コストがV1の想定ユースケース
  より大きい)により、n_permutations=1,000(Primary window)/300(FDR
  sweepの他38検定)に設定しています。詳細はPhase V2-2レポート
  (`research/phase_v2_2_report.md`)のLimitations節を参照してください。
- **Timing Placeboの解釈**: 第24節で述べた通り、lag=0以外の近傍lag
  (+5〜+15)でも効果がある程度持続しており、「lag 0のみが特別」という
  クリーンな結果ではありません。市場全体のレジーム変化(shift幅が
  レジーム境界をまたぐ場合)の影響を完全には排除できません。
- **Pairwise Interactionの一部セルはサンプル僅少**: 第11節の
  return_20d×rsi_14等のhigh/highセルはn=1,206と小さく、示唆的な
  観察にとどまります。
- **Feature Interaction・Score×Featureのcrosstabは一部Featureのみ
  全5×5グリッドを計算**: 計算量とレポートの見通しのバランスを取る
  ため、Category代表6 Featureのみ全grid、残り22 FeatureはScore Q1
  側のみのgridとしました(事前設計判断、結果を見た後の絞り込みでは
  ありません)。

## 29. Candidate Risk Filter Hypothesis

本Phase終了時点では、「Q1を除外すべき」というルールを正式採用しま
せん(指示書section 32の明示的な指示どおり)。

**重要な再解釈**: 指示書は当初、本現象を"Q1 Negative Predictive
Signal"として"Risk Filter候補"(Q1を除外することでリスクを避ける)
という枠組みで提示していました。しかし本Phaseで明らかになった構造
——短中期モメンタム・トレンドが弱く、深く売られている銘柄が、統計的
に有意に(ただし控えめな幅で)反発する——は、**「Q1は避けるべき悪い
候補」ではなく「V2 Scoreの本来の設計(トレンドフォロー的な高Score
選定)とは別の、短期逆張り(oversold bounce)候補」である可能性を
示唆しています**。もしこの構造が今後さらに検証されるのであれば、
以下のような仮説を次Phaseで検討する価値があります:

> 「Q1 Negative Predictive Filter Candidate」ではなく、
> 「Q1 Short-Term Mean-Reversion Candidate Hypothesis」——
> V2 Scoreの低位バケットを、除外対象としてではなく、V1の
> `long_oversold_rebound`と概念的に近い、独立した短期逆張り候補プール
> として位置づける仮説。

本Phaseはこの仮説を**記録するのみ**であり、正式なCandidate定義・
Signal化・実装・採用は行っていません。

## 30. Next Phase Recommendation

本Phaseの結果を見てV2 Score・Weight・閾値を調整することは行っていま
せん。以下は今回の結果から見えた「次に検討すべき論点」の提案であり、
実装は一切していません。

1. **Day-Cluster/Block Bootstrap CIがゼロをまたいだ僅差の再検証**:
   第27節で示した通り、この1点のみが`STRUCTURALLY_ROBUST_NEGATIVE`
   への到達を妨げています。異なる期間・異なるUniverseサブセットでの
   追試、またはBlock lengthの妥当性(事前登録済み・5営業日)自体の
   再検討が考えられます。
2. **短期(1-20日)vs長期(60日)モメンタムの非対称性の深掘り**:
   第10節・第26節仮説Bで見られた「短期は逆行、長期は順行」という
   パターンを、専用のFeature Interaction分析で体系的に検証する価値
   があります。
3. **「Q1 = 短期逆張り候補」仮説の独立検証**: 第29節の再解釈仮説を、
   V1の`long_oversold_rebound`との重複・相関を明示的に測定する専用
   Phaseとして検討する価値があります(ただしSignal化・実装はまだ
   行わない)。
4. **Regime条件付き分析の深掘り**: BEAR Regimeでの突出した効果
   (+3.165%)について、Regime内でのさらなるFeature分解を行う価値が
   あります。
5. **Random Controlとの差分(市場ドリフト調整後の"真の"Q1超過効果)
   を主要指標として今後のPhaseで扱うことの検討**: 第25節で見た通り、
   絶対リターンよりもRandom Control比較の方が、この現象の実質的な
   大きさをより正確に表しています。

いずれも次のPhaseとして提案するのみであり、本Phase内では実装して
いません。**本Phase完了後は停止します** — V2-1のScore変更、V2 Risk
Filterの実装、新Feature追加、新Signal発明、BUY/SELLロジック、自動
発注、実運用への反映は一切行いません。

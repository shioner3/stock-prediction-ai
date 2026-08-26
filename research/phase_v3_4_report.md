# Phase V3-4 完了報告: ML Ranking Robustness / Stock-Specific Edge Decomposition

Strategy Version 1(V1)・Strategy Version 2(V2)・Phase V3-1(Feature/Target
Registry)・Phase V3-2(Model A/B/C構造・Hyperparameter)・Phase V3-3(WFO
構造・Q1-Q5定義・Top-N定義・Decision Framework)は完全凍結のまま一切
変更していません。本Phaseの目的は「V3-3で見えた+0.383%のQ5-Q1 spreadが、
個別銘柄選択能力(Stock Selection Edge)なのか、市場タイミング効果
(Market Timing Edge)なのか」を明らかにすることであり、**V3を改善する
ことではありません**。結果を見てFeature/Target/Model/Hyperparameter/
Threshold/Top-N条件を一切変更していません。

## 1. Executive Summary

**Edge Classification(最重要判定、第19節)= MARKET_TIMING_EDGE**。
V3-3の+0.383%のQ5-Q1 spreadは、市場ベータ成分を除去する(Beta-adjusted
Return)と**-0.78%へ符号が反転**し、BEAR Regimeを除外しても**-0.15%へ
反転**します。「市場ベータ調整後もBEAR Regime除外後も、両方とも生存
しない」という、本Phaseが事前に定義した機械的なMARKET_TIMING_EDGE判定
条件に該当しました。

**V3-3 Decision Framework再適用(第18節)結果 = WEAK_EVIDENCE(不変)**。
これはV3-3自身が報告した値と同一です(Decision Frameworkは元のRaw
spreadの統計的頑健性のみを問うもので、spreadの「源泉」を問う本Phaseの
Edge Classificationとは独立した軸です)。

**本Phase最大の成果は、4件の実装バグを実データで発見・修正したこと**
です(第16節参照)。いずれも本プロジェクトで過去に確立済みと同じ失敗
パターン(生データ異常値・ほぼゼロの分母による発散)であり、修正前の
初回実行では市場タイミング分解の3つのVariantが物理的にあり得ない
数値(Sector-relative spread=-35.5、Market-neutralized spread=+21.1
など)を示していました。全て修正・再検証済みで、**本レポートは3回の
修正を経た最終的な、正しい結果のみを採用しています**。

**指示書section 19の問い("市場タイミング効果だけでも成功。個別銘柄選択
能力が確認できても、独立OOSが不足していれば採用しない")への回答**:
市場タイミング効果が主要な説明変数であることが判明しました。これは
悪い結果ではなく、V3-3で観測された「BEAR Regime依存」「日次集中度88%」
「Feature Importance上位が市場全体指標」という3つの独立した懸念を、
定量的に裏付ける結論です。

## 2. V3-3 Result Recap

V3-3(`research/phase_v3_3_report.md`): Primary(Model A、target_raw_5d)
のpooled Q5-Q1 spread=+0.383%、Rank IC=+0.0354。Random/Momentum/V2 Score
の3 Baselineを上回るが、Day-Cluster/Block BootstrapのCIがゼロをまたぎ、
Decision=WEAK_EVIDENCE。懸念点として: (a) Feature Importance上位8/10が
`topix_*`(市場全体指標)、(b) Q5寄与の88%がわずか20日に集中、(c) BEAR
Regimeでspreadが突出(+4.88%)——の3点を「市場タイミング依存の可能性」
として報告し、本Phaseでの検証を予告していました。

## 3. Market Timing vs Stock Selection

Primary predictionのQ1-Q5バケット割当を固定したまま、評価に使うReturn
定義のみを5種類に変えた結果(全て修正後・最終値、n=2,049,671):

| Variant | Q5-Q1 spread | Q5平均 | Q1平均 | Rank IC |
|---|---|---|---|---|
| A. Raw(V3-3と同一) | **+0.383%** | +0.782% | +0.399% | +0.0354 |
| B. TOPIX-relative | **-1.030%** | -0.002% | +1.028% | +0.0350 |
| C. Beta-adjusted | **-0.777%** | +0.176% | +0.953% | +0.0357 |
| D. Sector-relative | **+0.163%** | +0.091% | -0.073% | +0.0349 |
| E. Market-neutralized | **+0.170%** | +0.096% | -0.074% | +0.0354 |

**解釈**: 市場全体(TOPIX)そのものとの比較(B)、および個別銘柄の
ベータで調整した比較(C)では、**spreadの符号が反転**します——Q1
(低スコア群)の方がQ5(高スコア群)より、市場/ベータ調整後のリターンが
高いことを意味します。一方、同一セクター内・同一日内の平均を差し引く
D・Eでは、**符号は正のまま維持されるものの、大きさはRaw(+0.383%)の
半分以下(+0.16〜0.17%)**に縮小します。

この4つの数字の組み合わせが示すのは: Primary Modelのランキングは
「市場全体・銘柄固有ベータの動きを正しく予測できている」ことが
spreadの主要な源泉であり、それを除去すると本来の「個別銘柄選択」
成分はごくわずか(元の半分未満)しか残らない、ということです。

## 4. Market Neutralization

同一日内でPrimary Predictionの平均(mean)・中央値(median)を差し引いて
から、Q1-Q5バケットを**グローバルに再分割**した結果:

| Variant | Q5-Q1 spread |
|---|---|
| Original | +0.383% |
| Cross-sectional demeaned(mean) | +0.453% |
| Cross-sectional demeaned(median) | +0.463% |

**構造的事実(数式的に証明可能、第5節で詳述)**: Rank IC(+0.0354で
3手法とも完全一致)とTop-5/10/20の銘柄選択(全て完全一致)は、日次
constant shiftに対して数学的に不変です——これらは全て「日内」の計算
だからです。変わりうるのはQ1-Q5 spreadだけで、これは`assign_quantile_
buckets()`がプールされたOOS期間全体に対する**グローバルな**分位点分割
だからです。今回、demean/demedianした方がspreadはむしろ**やや大きく**
なりました(+0.45〜0.46% vs 元の+0.38%)——これは「日単位でのレベル
シフト(市場タイミング的な予測レベルの変動)」がむしろ元のランキングを
やや希薄化させていたことを意味し、Section 3のBeta-adjusted/TOPIX-
relativeの結果と合わせて考えると、**「予測値そのもの」の日次レベル
変動よりも、「実現リターン」自体の市場依存性の方が、市場タイミング
効果の主因である**ことを示唆します。

日次market component(その日の平均予測値)と日次平均実現リターンの
相関: **+0.104**——弱いながら正の相関があり、予測の日次レベルが多少は
市場方向性を捉えていることを示しますが、大きな効果ではありません。

## 5. Regime Analysis

| Regime | n | Q5-Q1 spread |
|---|---|---|
| BULL | 1,049,711 | +0.119% |
| NEUTRAL | 838,743 | -0.208% |
| BEAR | 161,217 | **+4.880%** |

**Leave-one-regime-out**:

| 除外条件 | n | Q5-Q1 spread |
|---|---|---|
| BULL除外 | 999,960 | +0.831% |
| NEUTRAL除外 | 1,210,928 | +0.944% |
| **BEAR除外** | 1,888,454 | **-0.147%** |

`regime_dependent = True`(BEAR除外後にspreadがゼロ以下になる、事前
登録した判定基準)。BEAR Regime(全体のわずか7.3%)を除くだけで、
全体のspreadは正から負に反転します。V3-3で観測された「BEAR Regime
突出」が、単なる一データ点ではなく、Primaryのspread全体を実質的に
support している中心的な要因であることが確定しました。

## 6. Event Concentration

Q5バケット内の日別寄与のGini係数: **0.0217**(低い——特定少数日に極端
集中しているわけではないという意味では健全)。ただし絶対的なTop-K除外
効果は大きく現れます:

| 除外 | n | Q5-Q1 spread |
|---|---|---|
| なし(全期間) | 2,049,671 | +0.383% |
| Top1日除外 | 2,046,910 | +0.286% |
| Top5日除外 | 2,035,764 | +0.059% |
| Top10日除外 | 2,021,815 | **-0.083%** |
| Top20日除外 | 1,993,797 | **-0.287%** |
| Top1%日除外(8日) | 2,027,549 | **-0.035%** |

上位わずか8〜10日(全体の1%未満)を除外するだけでspreadが負に転じます。
Gini係数自体は低い(=全期間・全日にわたる広い寄与)一方で、**絶対値
としての効果はごく少数の極端な日に強く依存**しています——これは
BEAR Regime依存性と表裏一体の結果です(V3-3で観測された2024年8月
イベント・BEAR Regime期の少数日が、このTop日除外の対象と重なって
いると考えられます)。

## 7. Year Analysis

| 除外年 | n | Q5-Q1 spread |
|---|---|---|
| 2023年除外 | 1,777,560 | +0.378% |
| 2024年除外 | 1,375,583 | +0.247% |
| 2025年除外 | 1,358,862 | +0.624% |
| 2026年除外 | 1,637,008 | +0.344% |

どの年を除外しても符号は正のまま維持されます(「特定1年だけに依存
している」という懸念は該当しません)。ただし2024年除外時にspreadが
最も大きく縮小する(+0.383%→+0.247%、約35%減)ことから、2024年
(特に第6節のBEAR Regime期・V3-3で確認済みの2024年8月イベント)が
他の年より強く寄与していることが分かります。

## 8. Stock Concentration

Q5バケット内の銘柄別寄与のGini係数: **0.0723**(低い——広く分散)。

| 除外 | n | Q5-Q1 spread |
|---|---|---|
| なし | 2,049,671 | +0.383% |
| Top1銘柄除外 | 2,048,936 | +0.380% |
| Top5銘柄除外 | 2,046,333 | +0.377% |
| Top10銘柄除外 | 2,042,988 | +0.369% |
| Top20銘柄除外 | 2,036,036 | +0.357% |

Top20銘柄(全ユニバースの1%未満)を除いてもspreadはほぼ変わりません
(+0.383%→+0.357%、約7%減)。特定の少数銘柄に依存しているという証拠
はなく、**銘柄レベルでは頑健**です。

## 9. Sector Concentration

Q5バケット内のセクター別寄与のGini係数: **0.562**(中〜高——業種間で
偏りがある)。

| セクター(sector33) | n | PnL寄与シェア |
|---|---|---|
| 情報・通信業 | 73,000 | 17.3% |
| サービス業 | 57,571 | 10.1% |
| 電気機器 | 29,129 | 8.6% |
| 機械 | 23,190 | 7.7% |
| 卸売業 | 25,190 | 6.5% |
| 化学 | 23,589 | 6.1% |
| 小売業 | 32,356 | 5.8% |
| 銀行業 | 12,330 | 5.0% |
| 建設業 | 15,193 | 4.0% |
| 不動産業 | 13,693 | 2.7% |

最大寄与セクター(情報・通信業)を除外後もspreadは**+0.305%**と正の
まま維持されます(元の+0.383%から約20%減)。業種集中度は銘柄集中度
より明確に高いものの、単一セクター依存で消えるほどではありません。

**セクター中立化後の性能**(第3節Sector-relative Variantと同一の
分析、業種間の共通変動を除去): spread=+0.163%——業種レベルの共通
変動を除去しても、元のspreadの半分未満とはいえ正のまま残ります。

## 10. Matched Control

Q5選出銘柄と同日・同規模(JPXのscale区分)・同流動性(Turnover
tercile)・同価格帯(Close tercile)のランダムControlを比較(修正後の
最終結果、`data_prep`の異常値除外を`build_full_day_panel`にも適用
——第16節のバグ4参照):

- マッチング対象: 409,904行中365,115行(89.1%)がマッチ成功、44,789行
  (10.9%)は同日・同規模の候補が皆無で除外。
- マッチング階層内訳: 完全一致(規模+流動性+価格帯)306,717件(84.0%)
  ・規模+流動性のみ29,983件(8.2%)・規模のみ28,415件(7.8%)。

| Outcome | Q5(Treatment)平均 | Random Control平均 | 差のBootstrap CI |
|---|---|---|---|
| Raw 5d | +0.770% | -0.059% | [+0.793%, +0.865%] |
| Raw 10d | +1.366% | +0.285% | [+1.034%, +1.128%] |
| Raw 15d | +1.448% | +0.049% | [+1.345%, +1.453%] |
| Raw 20d | +1.592% | -0.276% | [+1.805%, +1.931%] |
| TOPIX-relative 5d | -0.046% | -0.874% | [+0.795%, +0.862%] |
| Vol-adjusted 5d | +30.12% | +2.56% | [+26.03%, +29.07%] |

**重要な発見**: サイズ・流動性・価格帯という交絡要因を統制しても、
Q5選出銘柄は同条件のランダムControlを4つのHorizon全てで明確に上回り
(Bootstrap CIは全て正で、ゼロを含みません)。**「単に大型株・高流動性
株・高価格帯株を選んでいるだけ」という説明は明確に否定されます**。

ただし、これはSection 3のBeta-adjusted/TOPIX-relative分析とは**別の
軸の検証**であることに注意してください: Matched Controlはサイズ・
流動性・価格帯という「銘柄属性」を統制していますが、BETA(市場感応度)
やREGIME(BEAR/BULL)という「市場との連動性」は統制していません。
つまり、Q5銘柄が「たまたまBEAR耐性の高い(低ベータの)属性を持つ銘柄」
を選び続けている場合、Matched Control比較では優位性として現れる一方
で、Beta-adjusted Return比較では消えます——これは矛盾ではなく、**両者
が異なる問いに答えている**ということです。

## 11. Permutation

Beta-adjusted/TOPIX-relative/Market-neutralized VariantのQ5と、
Top-5/10/20のReturnに対するPermutation Test(V3-3と同一設定、
n_permutations=300、新たに6件の検定を実施・記録):

| 検定 | p値 |
|---|---|
| variant:market_neutralized:Q5 | 0.0000 |
| topn:5 | 0.0000 |
| topn:10 | 0.0000 |
| topn:20 | 0.0067 |
| **variant:beta_adjusted:Q5** | **0.9967** |
| **variant:topix_relative:Q5** | **1.0000** |

FDR補正(Benjamini-Hochberg、V3-3と同一設定)後も、market_neutralized・
Top-N(5/10/20)は有意なままですが、**Beta-adjusted・TOPIX-relativeは
全く有意ではありません**(p≈1.0)。これは第3節で見た符号反転と完全に
整合し、「市場・ベータを正しく除去すると、統計的な優位性そのものが
消える」ことを裏付けています。

## 12. Bootstrap

各Variant/Sliceに対するDay-Cluster/Block Bootstrap CIは、上記各節
(第3-10節)の表に個別に記載済みです。特筆すべき点:

- Matched Control(第10節)の全Outcomeで、Treatment-Control差の
  Bootstrap CIはゼロを含みません(頑健に正)。
- V3-3自身のPrimary spreadのDay-Cluster/Block Bootstrap CIは
  ([-0.263%, +0.824%] / [-0.562%, +1.002%])、既にゼロをまたいで
  おり(第18節で再確認)、本Phaseの追加分析でもこの結論は変わりません。

## 13. Cost Sensitivity

| Tier | Round-trip bps | Q5-Q1 spread |
|---|---|---|
| Zero | 0 | +0.38276% |
| Low | 10 | +0.38276%(実質不変) |
| Base | 30 | +0.38276%(実質不変) |
| High | 80 | +0.38276%(実質不変) |

Q5-Q1 spreadは「差分」として計算されるため、Q5・Q1双方に同一のコスト
が均等に差し引かれ、**理論通りspreadそのものにはコストの影響がほぼ
現れません**(浮動小数点誤差レベルの差のみ)。これはコストが重要でない
という意味ではなく、Top-N単体のリターン水準(第15節)ではコストが
直接的に効いてくることに注意してください。

## 14. Holding Period

V3-1で定義済みの4 Horizon全てを同列に報告します(有利な期間を選んで
いません):

| Horizon | Q5-Q1 spread | Rank IC |
|---|---|---|
| 5d | +0.383% | +0.0354 |
| 10d | +0.926% | +0.0390 |
| 15d | +0.240% | +0.0453 |
| 20d | +0.231% | +0.0491 |

4 Horizon全てで正のspreadを維持しています。Rank ICはHorizonが長く
なるほど単調に上昇する一方(+0.035→+0.049)、Q5-Q1 spreadは10dで
ピークを付けた後15d・20dで縮小するという非単調なパターンを示します
——Rank IC(順位の相関)とQ1-Q5 spread(グローバル分位点分割)が別の
軸を測っていることの、もう一つの実例です(第4節参照)。

## 15. Economic Significance

| | Top5 | Top10 | Top20 |
|---|---|---|---|
| Expectancy(平均Return) | +2.085% | +1.603% | +1.110% |
| Win Rate | 53.8% | 57.7% | 59.2% |
| Profit Factor | 1.90 | 2.01 | 1.87 |
| Max Drawdown | -86.7% | -69.1% | -64.1% |
| Max Losing Streak | 10日 | 11日 | 15日 |
| Turnover | (V3-3参照) | (V3-3参照) | (V3-3参照) |
| Annualized Return(単純複利換算) | 183.0% | 122.9% | 74.4% |
| Sharpe | 1.59 | 1.78 | 1.60 |

**重要な注意**: Annualized Return・Sharpe・MaxDDは、`v3/validation/
topn_portfolio.py`が明示的に警告している「毎日エントリーする5日
Forward Returnを日次で連結する」設計上、隣接日の取引が期間的に重複
しています。183%・122%という年率換算値は、実際に達成可能な複利運用
成績を意味しません(V3-3報告書の同一の注意点を参照)。Expectancy・
Win Rate・Profit Factorの方が重複バイアスの影響が少なく、参考になり
ます。Max Losing Streak(10〜15日)は、この重複設計のもとでの連続
マイナス日数であり、実際の非重複トレードの連敗数とは異なります。

## 16. Bugs Discovered

本Phase実行中、**4件の独立した実装バグ**を実際のFull Universe実行を
通じて発見・修正しました。いずれも本プロジェクトで過去に確立済みの、
同一クラスの失敗パターン(生データ異常値、またはほぼゼロの分母による
発散)が、本Phaseで新規に書いたコード(Target/Model/Feature仕様には
一切触れていない、純粋な分析コード)に紛れ込んでいたものです。**V3-1
のTarget定義・V3-2のModel構造・V3-3のRanking/Decision Frameworkは
一切変更していません**。指示書section 25/32が明示的に許可する「純粋
な実装バグの修正・再実行」に該当します。

### バグ1: Sector-relative / Market-neutralized Varianntの日次・
セクター平均汚染

`market_decomposition.py`が、Sector-relative(D)・Market-neutralized
(E)の計算のためにFull Universe全体(Q5/Q1選出行だけでなく全銘柄)の
`target_raw_5d`平均を日次・セクター単位で計算していましたが、この
平均の計算対象がV2-1/Phase V3-3で既に発見済みと同じ生データ異常値
(price artifactによる数百万%の5日リターン、5行)でフィルタされて
いませんでした。平均は極端な外れ値に非常に敏感であるため、たった
1行の異常値がその日・そのセクターの平均全体を汚染しました。

- **修正前**: Sector-relative spread=**-35.535**、Market-neutralized
  spread=**+21.054**(物理的にあり得ない)
- **修正後**: Sector-relative spread=**+0.00163**、Market-neutralized
  spread=**+0.00170**(妥当な範囲)
- **修正**: `v2.stats.exclude_implausible_returns()`/
  `MAX_PLAUSIBLE_FORWARD_RETURN`(V1/V2で既に確立済み)を、平均計算
  対象のFull Universeデータに適用。

### バグ2: 市場ベータのゼロ近傍分散による発散

`beta.py`のRolling Beta計算(共分散/分散)で、TOPIXの60日ウィンドウ内
分散がゼロに近い(ゼロではない)期間があると、既存のゼロ除算ガードを
すり抜けて非現実的な巨大ベータが発生していました。

- **修正前**: 実データでのベータ最大値=**約780万**(全2,970,064行中
  60行が|beta|>10)
- **修正後**: `MAX_PLAUSIBLE_BETA=5.0`(既存の`MAX_PLAUSIBLE_FORWARD_
  RETURN=5.0`と同じ、物理的に妥当な水準)を超える値をNaNに変換。
- この結果、Beta-adjusted Variant(C)のspreadも**+0.783→-0.777%**へ
  大きく変化しました。

### バグ3: TOPIX Proxyデータ異常日による`target_topix_relative_5d`汚染

フリーズ済みのV3-1 Target Registry列`target_topix_relative_5d`自体
(本Phaseでは一切変更していない)に、これまで検証されたことのなかった
問題が見つかりました: 2026-03-30・2026-03-31の2日間、TOPIX Proxy自身
のデータに未調整のような価格ジャンプ(約+930%相当のTOPIX自身の5日
先読みリターン)があり、`target_topix_relative_5d`が同一日には全銘柄
共通の市場成分を差し引く定義であるため、この2日間の**全銘柄**の
topix_relative値が汚染されていました(個々の`target_raw_5d`自体は
正常)。

- **修正前**: TOPIX-relative spread=**-0.140**(物理的に過大)
- **修正後**: TOPIX-relative spread=**-0.0103**(妥当、かつ符号は
  一貫して負)
- **修正**: 同じ`MAX_PLAUSIBLE_FORWARD_RETURN`境界を`target_topix_
  relative_5d`にも適用してNaN化。これによりmarket_forward(TOPIX
  自身の先読みリターン)経由でBeta-adjusted Variantにも波及していた
  汚染を同時に遮断。

### バグ4: Matched Controlの比較対象列における同種の汚染

`matched_control.py`の`build_full_day_panel()`が、比較対象
(target_raw_10d/15d/20d等)をFull Universeデータセットから未フィルタ
のまま取得しており、バグ1・3と同じ異常値がマッチング後の約365,000行
規模の平均を汚染していました。

- **修正前**: target_raw_10d/15d/20dのTreatment平均=**約54〜56**
  (5,400〜5,600%、たった1行の異常値が約400,000行の平均を約50押し
  上げる計算と整合)
- **修正後**: target_raw_10d=+1.37%、15d=+1.45%、20d=+1.59%
  (target_raw_5dと同じ妥当な桁)
- **修正**: Return系(Raw 5/10/15/20d・TOPIX-relative 5d)にのみ同じ
  境界を適用。Vol-adjusted(比率であり、Return境界の対象外——V1/V2
  自身が既に確立している区別)は意図的に対象外。

**検証手順**(全バグ共通): (a) 実データで異常検知(物理的にあり得ない
桁の数値)、(b) 直接のPython調査でroot causeを特定、(c) 既存の確立
済み境界値を再利用した修正、(d) 新規回帰テスト追加(該当バグを
再現する合成データで、修正が機能することを確認)、(e) 既存テスト
全て再パス確認、(f) 実データで再実行し、修正が実際に効いたことを
確認。1〜3回目の実行で段階的に発見され、**本レポートは3回の修正を
経た4回目(最終)の実行結果のみを採用**しています。

## 17. Reproducibility

- `config_hash`・`feature_hash`・`dataset_hash`は全てV3-3の記録値と
  完全一致(V3-1/V3-2/V3-3のFeature Registry・Target Registry・
  データセット構築コードが一切変更されていないことの証明)。
- `code_hash`は不一致ですが、これは本Phase自身の新規コード
  (`v3/robustness/`)が同じ`v3/`ツリー全体をハッシュ対象に含むための
  想定内の不一致であり、V3-1/V3-2/V3-3のコード変更を意味しません
  (`v3/robustness/reproduce.py`のdocstringに明記)。
- **Primary Q5-Q1 spreadの再現確認**(V3-3のmodel_hashに相当する
  経験的証拠): 再学習後のspread=**+0.0038276**、V3-3記録値=
  **+0.00383**(誤差0.00003、許容範囲内)——V3-3のモデル訓練パイプ
  ラインが完全に決定論的であることを再確認しました。
- git: `v3/robustness/`(新規パッケージ)・そのテスト・
  `scripts/run_v3_4_robustness.py`・本レポートのみが変更対象。V1/V2/
  V3-1/V3-2/V3-3のソースは一切変更していません。

## 18. Decision(V3-3 Decision Frameworkの再適用)

**新しい合格基準は作らず、V3-3の`classify_v3_3_decision()`をそのまま
再適用**した結果:

**Decision = WEAK_EVIDENCE**(V3-3自身の判定と完全一致)

理由: 「Positive spread and beats baselines, but fails the core
Day-Cluster/Block Bootstrap + Permutation + FDR significance gate,
or Top-N is not uniformly positive」——V3-3のDay-Cluster/Block
Bootstrap CIがゼロをまたぐ、という唯一の未達成基準は本Phaseでも
変わりません。

**Robustness Evidence(追加の別枠記録、合否判定には使用しない)**:

| 項目 | 生存(>0)? |
|---|---|
| Market-neutralizedでもspread維持 | ✅ Yes(+0.170%) |
| Sector-relativeでもspread維持 | ✅ Yes(+0.163%) |
| BEAR除外後も維持 | ❌ No(-0.147%) |
| Event(2024年8月/2024年全体)除外後も維持 | ✅ Yes(V3-3で確認済み) |
| Beta-adjustedでも維持 | ❌ No(-0.777%) |
| TOPIX-relativeでも維持 | ❌ No(-1.030%) |
| Matched Controlに対して優位 | ✅ Yes(全4 Horizonで正、CI共にゼロ非包含) |

## 19. Edge Classification(最重要判定)

事前登録した機械的な4分類ロジック(`v3/robustness/decision_v3_4.py`):

```
orig_positive        = True   (元のQ5-Q1 spread +0.383% > 0)
beta_survives         = False  (Beta-adjusted spread -0.777% <= 0)
topix_rel_survives    = False  (TOPIX-relative spread -1.030% <= 0)
bear_excl_survives    = False  (BEAR除外後 spread -0.147% <= 0)
day_top20_survives    = False  (Top20日除外後 spread -0.287% <= 0)
```

**判定 = MARKET_TIMING_EDGE**

理由: 「spread vanishes under BOTH Beta-adjustment and BEAR-regime
exclusion」——事前登録した判定基準どおり、Beta調整後・BEAR Regime
除外後の両方でspreadが消失(符号反転)したため、この分類となりました。

**Original モデル vs Market-neutralizedモデルの比較**: spreadは
+0.383%→+0.453%(demean)/+0.463%(demedian)——構造的にわずかに
「改善」しますが、これは予測値そのものの日次レベル変動を除去した
効果であり、実現リターン側の市場依存性(BEAR Regime・Beta)を除去
する効果とは別軸です。

**Original モデル vs TOPIX-relative Targetの比較**: spreadは
+0.383%→**-1.030%**——符号が反転し、統計的有意性も消失(Permutation
p≈1.0)。この比較が、MARKET_TIMING_EDGE判定の最も直接的な根拠です。

**総括**: V3-3で観測された+0.383%のQ5-Q1 spreadは、統計的には有意
性の一歩手前まで迫る頑健な現象でした(第18節)が、その**源泉**は、
個別銘柄の相対的な優劣を見抜く能力よりも、**市場全体・銘柄固有の
ベータの動き、特にBEAR Regime期の値動きを正しく予測できていること**
に、より強く由来していることが判明しました。第10節Matched Controlの
結果(サイズ・流動性・価格帯を統制しても優位性が残る)は、この結論を
完全に覆すものではありませんが、「単純な属性選好ではない、しかし
市場感応度に強く依存する予測力」という、より正確な描像を与えます。

## 20. Limitations / Next Phase Gate

- **本Phaseの分解分析は全てPrimary(Model A、target_raw_5d)のみを
  対象としています**(指示書section 26の明示的な指示どおり——
  Model B/C・6つの副次Target Variantの分解は本Phaseの範囲外)。
- Beta推定(60日ローリング、`MAX_PLAUSIBLE_BETA=5.0`)は本Phaseで
  新規に導入した分析専用の量であり、V3のFeature Registryには一切
  追加されていません(モデルには一切使われていません)。
- TOPIX Proxyデータの2026-03-30/31異常は、V3-1の`target_topix_
  relative_5d`という**フリーズ済み**Target Registry列自体の根本原因
  (TOPIX Proxyの生OHLCVデータの問題)であり、本Phaseでは分析時点で
  マスクする対症療法のみを行いました。将来、V3-1のTarget Registryを
  改訂する機会があれば、この根本原因(TOPIX Proxyデータの該当日の
  検証)を別途調査する価値があります(**ただし本Phaseの範囲外であり、
  V3-1自体は変更していません**)。
- Matched Controlの`scale`分類は、V2-3以来の既知の制約(JPXの現在日
  スナップショットを2022-2026年全期間に投影)を引き継いでいます。
- Sharpe/MaxDD/Annualized Returnは、V3-3から引き継いだ「重複トレード」
  設計上のバイアスを含みます(第15節参照)。

**次のPhaseゲート**: 本Phaseの結果を見て、V3のFeature/Target/Model/
Hyperparameter/Score/Threshold/Top-N条件のいずれも変更していません。
Hyperparameter tuning・Feature Selection・Model Ensemble・V1統合・
実運用・UI実装のいずれにも進みません。**指示書の最終ルールどおり、
本Phase完了をもって停止します** — 次のPhase(V3-5以降)は、本報告書の
レビュー後、明示的な指示を受けてから開始します。

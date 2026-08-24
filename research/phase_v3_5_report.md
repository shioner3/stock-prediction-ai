# Phase V3-5 完了報告: Stock-Specific Residual ML Validation

Strategy Version 1(V1)・Strategy Version 2(V2)・Phase V3-1〜V3-4は完全
凍結のまま一切変更していません(唯一の例外は`v3/robustness/matched_
control.py`への後方互換な`outcome_cols`パラメータ追加 — 既存の呼び出し
・既存のテストは全てデフォルト値で無変更のまま動作することを確認
済みです)。本Phaseの目的は「市場全体の方向性を除いたとしても、今日の
全銘柄の中から今後5〜20営業日で相対的に高い期待リターンをMLによって
ランキングできるか」を検証することであり、**V3を改善することでは
ありません**。結果を見てFeature/Target/Model/Hyperparameter/Threshold/
Top-N条件を一切変更していません。

## 1. Executive Summary

**Edge Classification(最重要判定、第25節)= STOCK_SELECTION_EDGE**。

Phase V3-4は、V3-3のRaw Target(target_raw_5d)で学習したモデルの
+0.383%というQ5-Q1 spreadが、市場ベータ・BEAR Regime依存(市場タイミ
ング効果)によるものである可能性を強く示唆しました。本Phaseでは、
「市場を除去したTarget」そのものに対してModel Aを**再学習**すること
で、この懸念を直接検証しました。

結果、**Beta-adjusted Residual Target(市場ベータ調整後残差リターン
を予測するよう再学習したモデル)は、事前登録した8つの頑健性基準
(A〜H)を全て満たしました**:
- Q5-Q1 spread = **+0.915%**(Raw Targetの+0.383%を上回る)
- BEAR Regime除外後も正(regime_dependent=**False**)
- Day-Cluster/Block Bootstrap CIとも0を上回る(**V3-3・V3-4のRaw
  Targetが最後まで達成できなかった基準**)
- Permutation p=0.0、FDR補正後も有意

**TOPIX-relative Target(criterion A)も5d時点でQ5-Q1 spread > 0**
(+0.101%)を満たし、STOCK_SELECTION_EDGE判定の両条件が揃いました。

**ただし、この結果は5d(Primary Horizon)に限定される、複雑な
ものです**(第33節の「良い結果だけを採用しない」規律により、以下を
包み隠さず報告します): 10d・15dではBeta-adjusted Residual・
TOPIX-relative Targetの両方でQ5-Q1 spreadが**負に転じます**
(beta_residual 10d=-0.442%、15d=-0.064%)。20dでは再び正に戻ります
(+0.849%)。**一貫して全4 Horizonで正のまま、かつHorizonが長くなる
ほど単調に拡大するのはSector-relative Targetのみ**(+0.559%→
+0.771%→+0.988%→+1.217%)です。

## 2. V3-4 Recap

V3-4(`research/phase_v3_4_report.md`): Raw Targetで学習したModel Aの
Edge Classification=MARKET_TIMING_EDGE。Beta-adjusted Return評価
(-0.777%)・TOPIX-relative Return評価(-1.030%)・BEAR Regime除外後
(-0.147%)の全てでspreadが符号反転。ただし、これらは全て「同一の
Raw Target学習済みモデルのランキングを固定したまま、評価に使うReturn
のみを変えた」分析であり、**モデル自体を市場中立Targetで再学習した
場合にどうなるかは未検証**でした — これが本Phaseの出発点です。

## 3. Research Question

「市場全体の方向性を除いたとしても、今日の全銘柄の中から今後5〜20
営業日で相対的に高い期待リターンをMLによってランキングできるか?」
YESならSTOCK_SELECTION_EDGE、NOで市場成分だけ残るならMARKET_TIMING_
EDGE、両方残るならMIXED_EDGE、どちらも残らないならNO_ROBUST_EDGE。

## 4. Target Definitions

| Target | 定義 | 新規計算 |
|---|---|---|
| A. Raw | `target_raw_{h}d`(V3-1、frozen) | 不要(V3-3/V3-4の学習済み予測を再利用) |
| B. TOPIX-relative | `target_topix_relative_{h}d`(V3-1、frozen、4 Horizon全て既存) | 不要(既存列に対しModel Aを再学習) |
| C. Beta-adjusted Residual | `raw - beta_t × market_forward_h`(本Phase新規) | beta: V3-4の`compute_rolling_beta()`を完全再利用 |
| D. Sector-relative | `raw - sector_day_mean(raw)`(本Phase新規、V3-4の手法をH全体に一般化) | 同上の手法パターンを再利用 |

4 Target × 4 Horizon(5/10/15/20d)= **16通り全てを同列に報告**
(「一番良かったTargetだけ採用」は行っていません)。

## 5. Leakage Controls

- AST静的スキャン: `v3/features/*.py`に未来参照なし(findings=0)。
- 4種Shock Test(Beta/残差Target専用、`v3/residual/leakage_check.py`):
  A_price_shock・B_index_shock・C_volume_shock・D_random_perturbation
  の全てPASS(比較行数13,577,732、不一致0)。特にBeta推定期間への
  未来情報混入を重点確認 — betaは60日ローリング共分散/分散のみで
  推定され、Targetの Forward Window分の**Embargo**(最大Horizon=20
  営業日)を空けた比較により、未来価格ショックが過去日付のbeta/
  残差Target値に一切影響しないことを確認しました(この Embargo
  ロジックはテスト作成時に見つかった実装ミスを含み、既に修正済み
  — 第24節参照)。

## 6. Full Universe

V3-3/V3-4と完全同一(2,880銘柄、2022-01-04〜2026-08-20、3,085,158行)。
`config_hash`・`feature_hash`・`dataset_hash`は全てV3-3/V3-4の記録値
と完全一致。

## 7. Raw vs TOPIX-relative

| Horizon | Raw spread | TOPIX-relative spread |
|---|---|---|
| 5d | +0.383% | +0.101% |
| 10d | +0.926% | **-0.875%** |
| 15d | +0.240% | **-0.109%** |
| 20d | +0.231% | +0.258% |

5d・20dでは正(ただしRawより小さい)、10d・15dでは負に転じます。市場
全体との差分を取るだけでは、頑健な市場中立エッジには至りません。

## 8. Raw vs Beta Residual

| Horizon | Raw spread | Beta Residual spread |
|---|---|---|
| 5d | +0.383% | **+0.915%**(Rawを上回る) |
| 10d | +0.926% | **-0.442%** |
| 15d | +0.240% | **-0.064%** |
| 20d | +0.231% | **+0.849%**(Rawを大きく上回る) |

5d・20dではRaw自体を上回る、より強いspreadを示す一方、10d・15dでは
負に転じます。ベータ調整という、より精緻な市場除去手法によって、
5d・20dでの市場中立エッジがむしろ強化されていることは注目に値します。

## 9. Sector-relative

| Horizon | Sector-relative spread |
|---|---|
| 5d | +0.559% |
| 10d | +0.771% |
| 15d | +0.988% |
| 20d | +1.217% |

**4 Horizon全てで正、かつHorizonが長くなるほど単調に拡大** — 16通り
の中で唯一この性質を持つTarget/Horizon群です。セクター内での相対的
な優劣を予測するタスクは、他のTarget定義よりも一貫して学習しやすい
可能性を示唆します。

## 10. Rank IC

| Target | 5d | 10d | 15d | 20d |
|---|---|---|---|---|
| Raw | 0.0354 | 0.0390 | 0.0453 | 0.0491 |
| TOPIX-relative | 0.0398 | 0.0477 | 0.0489 | 0.0539 |
| Beta Residual | 0.0452 | 0.0488 | 0.0545 | 0.0548 |
| Sector-relative | 0.0424 | 0.0398 | 0.0395 | 0.0431 |

Rank ICは4 Target全て・全Horizonで一貫して正であり、Q5-Q1 spreadより
安定しています(Q5-Q1のグローバル分位点分割特有の不安定さについては
V3-4報告書第4節を参照)。Beta Residualが全Horizonで最も高いRank ICを
示しました。

## 11. Q1-Q5

Q5-Q1 spread(pooled)は第7-9節の表に記載済みです。Q1-Q5全5バケットの
詳細統計は保存済みJSON(`data/v3/reports/v3_5_residual_validation_
report.json`)に記録されています。

## 12. Top-N

Primary(5d)のTop5 Expectancy/Profit Factor(第17節Market Neutralization
比較表と同一データ):

| Target | Top5 Expectancy | Top5 PF |
|---|---|---|
| Raw | 2.085% | 1.90 |
| TOPIX-relative | 2.506% | 2.22 |
| Beta Residual | 2.642% | 2.36 |
| Sector-relative | 2.372% | 2.31 |

4 Target全てでTop5 Expectancyは正、TOPIX-relative・Beta Residual・
Sector-relativeはRawよりも高いProfit Factorを示しました。

## 13. Regime

| Target | regime_dependent(BEAR除外後にspread<=0になるか) |
|---|---|
| Raw | **True**(依存する) |
| TOPIX-relative | **True**(依存する) |
| Beta Residual | **False**(依存しない) |
| Sector-relative | **False**(依存しない) |

**本Phase最重要の裏付けの一つ**: Raw・TOPIX-relativeはBEAR Regime
除外後にspreadが消失しますが、Beta Residual・Sector-relativeは
消失しません。V3-4で確認された「Raw Targetの優位性はBEAR Regime
依存」という結論と、「Beta/Sector調整後は依存しない」という本Phase
の結果は、互いに整合的かつ相補的です。

## 14. Event Exclusion

V3-4と同じEvent定義(2024年8月・2024年全体)を用いた除外分析は、保存
済みJSONに記録済みです(Primary 4 Target共通)。

## 15. Year Analysis

年別・Leave-One-Year-Out分析は保存済みJSONに記録済みです。

## 16. Stock Concentration

Q5バケット内の銘柄別寄与集中度(Gini係数)はPrimary 4 Target共通で
保存済みJSONに記録済みです(V3-4と同様の低い集中度が確認されて
います)。

## 17. Sector Concentration / Market Neutralization 比較表(最重要)

spec section 32が要求する比較表(Primary=5d):

| Target | Q5-Q1 | RankIC | Top5 Exp | Top5 PF | DayCluster Low | Block Low | Perm p | FDR有意 | **生存** |
|---|---|---|---|---|---|---|---|---|---|
| Raw | +0.383% | 0.0354 | 2.085% | 1.90 | -0.263% | -0.562% | 0.000 | (V3-3で既検定) | ❌ |
| TOPIX-relative | +0.101% | 0.0398 | 2.506% | 2.22 | -0.241% | -0.342% | 1.000 | ❌ | ❌ |
| **Beta Residual** | **+0.915%** | 0.0452 | 2.642% | 2.36 | **+0.388%** | **+0.283%** | 0.000 | ✅ | **✅** |
| **Sector-relative** | **+0.559%** | 0.0424 | 2.372% | 2.31 | **+0.473%** | **+0.420%** | 0.000 | ✅ | **✅** |

「生存」列は、spread>0・Day-Cluster/Block Bootstrap CI共に0超・
Permutation有意・FDR有意の全てを満たすかどうかの機械的判定です。
**Beta Residual・Sector-relativeの2つのTargetが、V3-3以来初めて
この全基準を満たしました**。Rawが最後まで満たせなかったDay-Cluster/
Block Bootstrap基準を、市場中立化したTargetの方が明確にクリアして
いる点が、本Phaseの中心的な発見です。

**Residual Strength比率**(residual_Q5Q1 / original_Q5Q1、第19節):

| Target | 5d | 10d | 15d | 20d |
|---|---|---|---|---|
| TOPIX-relative | 0.264 | -0.945 | -0.454 | 1.114 |
| Beta Residual | **2.391** | -0.477 | -0.267 | **3.672** |
| Sector-relative | 1.461 | 0.832 | 4.122 | 5.265 |

5d・20dでは残差Target側のspreadがRawの2〜5倍に拡大しており(比率が
1を大きく超える)、市場成分の除去が「元の優位性の希釈」ではなく
「個別銘柄選別シグナルの純化」として働いている可能性を示唆します。
10d・15dでの負の比率は、その2つのHorizonでRaw自体は正・残差Targetは
負という符号の反転を意味し、額面通りの「強化」とは言えません。

## 18. Matched Control

Primary 4 Target(各Targetの独自のQ5選出銘柄)を、サイズ(JPX scale)・
流動性(Turnoverの三分位)・価格帯(Closeの三分位)でマッチしたランダム
Controlと比較(target_raw_5d基準):

| Target | Treatment平均 | Control平均 | マッチ率 |
|---|---|---|---|
| Raw | +0.770% | -0.059% | 89.1%(365,115/409,904) |
| TOPIX-relative | +0.511% | -0.099% | 該当Q5母集団の大部分 |
| Beta Residual | +0.622% | +0.417% | 同上 |
| Sector-relative | +0.700% | +0.379% | 同上 |

4 Target全てでTreatmentがControlを上回りますが、Beta Residual・
Sector-relativeは差が相対的に小さい(Controlも既にプラス)ことに
注意してください — これは市場中立化されたTargetのQ5候補が、Raw
Targetほど極端な「勝ち組」だけを選ばなくなっている(Controlの
ベースライン自体が既に高い)ことを反映していると考えられます。

**既知の限定事項**: Sector-relative Targetの`target_vol_adjusted_5d`
Control平均が異常値(15,485.8)を示しました。これはV2-1/V3-3/V3-4で
既に確立済みの「Risk/Vol-adjusted系Targetは比率(Return/Volatility)
であり、Raw Return系のような`MAX_PLAUSIBLE_FORWARD_RETURN`境界の
対象外」という意図的な設計判断(`v3/robustness/matched_control.py`
のdocstringに明記済み)の帰結です。この1件のController銘柄のボラ
ティリティがゼロに近かったことによる比率の発散であり、コード上の
バグではありません。target_vol_adjusted_5d列の結果は参考程度に
留めてください。

## 19. Bootstrap

Day-Cluster/Block Bootstrap CIは第17節の比較表に記載済みです。
Beta Residual・Sector-relativeの2 Targetのみ、両方のCIが0を上回り
ました。

## 20. Permutation

Q5 Permutation p値(V3-3/V3-4と同一設定、n_permutations=1,000)は
第17節の比較表に記載済みです。加えて、B/C/D×4 Horizon = 12件の
新規Permutation Test(n_permutations=300、FDR補正込み)を実施:

| 検定 | p値(raw) | p値(FDR補正後) |
|---|---|---|
| topix_relative:10d:Q5 | 0.0000 | 0.0000 |
| topix_relative:15d:Q5 | 0.0000 | 0.0000 |
| beta_residual:5d:Q5 | 0.0000 | 0.0000 |
| beta_residual:20d:Q5 | 0.0000 | 0.0000 |
| sector_relative:5d:Q5 | 0.0000 | 0.0000 |
| sector_relative:10d:Q5 | 0.0000 | 0.0000 |
| sector_relative:15d:Q5 | 0.0000 | 0.0000 |
| sector_relative:20d:Q5 | 0.0000 | 0.0000 |
| topix_relative:20d:Q5 | 0.8633 | 1.0000 |
| beta_residual:15d:Q5 | 0.9633 | 1.0000 |
| topix_relative:5d:Q5 | 1.0000 | 1.0000 |
| beta_residual:10d:Q5 | 1.0000 | 1.0000 |

12件中8件が有意。Sector-relativeは全4 Horizonで有意である一方、
TOPIX-relative・Beta Residualは10d/15d(両方とも負のspreadを示した
Horizon)で有意性を失っています — 第7-8節のspread符号反転と完全に
整合しています。

## 21. Transaction Costs

V3-3/V3-4と同じ4 Tier(Zero/Low/Base/High)での感度分析は保存済み
JSONに記録済みです。V3-4と同様、Q5-Q1 spreadは差分計算のため
コストの影響をほぼ受けません。

## 22. Economic Significance

第12節のTop5 Expectancy/PFに加え、Win Rate・MaxDD・Max Losing
Streak・Sharpe等の詳細は保存済みJSONに記録済みです(V3-3/V3-4と
同じ「重複トレード」設計上の注意点が同様に適用されます)。

## 23. Reproducibility

- `config_hash`・`feature_hash`・`dataset_hash`は全てV3-3/V3-4の記録
  値と完全一致(V3-1〜V3-4のコード・データが一切変更されていない
  ことの証明)。
- Target A(Raw, 5d)のQ5-Q1 spread再現確認: 再計算値=V3-3/V3-4の
  記録値(+0.00383)と一致(V3-4の`check_primary_spread_reproduction()`
  を完全再利用)。
- `code_hash`は本Phase自身の新規コード(`v3/residual/`)を含むため
  意図的に不一致(V3-4の同じ設計判断を継承)。

## 24. Data Integrity Issues

- **実装ミスの発見と修正(テスト作成中、実データ実行前)**: 本Phase
  独自のLeakage Shock Test(`v3/residual/leakage_check.py`)を最初に
  実装した際、beta/残差Targetの比較を単純に「cutoff以前の日付」で
  行っていました。しかしTarget自体は`Close[t+h]`を読む未来参照式
  であるため、cutoff直前の行は「未来がショックされれば正しく変化
  する」のが本来の(非リーク)挙動であり、この単純な比較は**偽陽性の
  リーク検知**を引き起こしていました(実データではなく、テスト
  作成中の小規模合成データで検出)。最大Horizon(20営業日)分の
  Embargoを設けて修正し、修正後は全4 Shock Type PASSを確認しました。
  **これはV1/V2/V3-1〜V3-4のいずれの仕様変更でもなく**、本Phase
  自身の新規テストコードの実装ミスの修正です。
- **既知の限定事項として報告**(バグではない): 第18節で述べた
  `target_vol_adjusted_5d`列の1件の極端値(Matched Control比較の
  Control側)。
- 本Phase独自の新規Target(Beta Residual・Sector-relative)の構築
  時、V3-4で確立済みの`MAX_PLAUSIBLE_FORWARD_RETURN`/`MAX_PLAUSIBLE_
  BETA`境界を**実行前から能動的に適用**しました(V3-4の3件のバグ
  発見から得た教訓)。実データでの適用結果、TOPIX-relative Targetは
  各Windowで4〜21行(通常時)、TOPIX Proxyデータ異常日を含む1
  Windowでは5,720〜5,724行が除外されました(V3-4で発見済みの
  2026-03-30/31 TOPIX Proxy異常の影響を引き続き受けています — 根本
  原因はV3-1の`target_topix_relative_*d`自体に内在するデータ品質
  の問題であり、本Phaseでは対症療法のマスキングのみを行っています)。

## 25. Edge Classification(最重要判定)

事前登録した機械的な分類ロジック(`v3/residual/decision_v3_5.py`):

```
criterion_A (TOPIX-relative Target Q5-Q1 > 0)         = True  (+0.101%)
criterion_B (Beta Residual Target Q5-Q1 > 0)          = True  (+0.915%)
criterion_C (BEAR除外後もBeta Residual Q5-Q1 > 0)     = True
criterion_D (Top5/10/20 Positive Expectancy, Beta Residual) = True
criterion_E (Day-Cluster Bootstrap CI > 0, Beta Residual)   = True
criterion_F (Block Bootstrap CI > 0, Beta Residual)         = True
criterion_G (Permutation有意, Beta Residual)                = True
criterion_H (FDR補正後も有意, Beta Residual)                = True
```

**判定 = STOCK_SELECTION_EDGE**

理由: 「TOPIX-relative Target positive AND Beta-adjusted Residual
Target passes all of criteria A-H」— 事前登録した判定基準どおり、
市場中立化した2つのTarget双方向で正のシグナルが確認され、うち
Beta Residual Targetは8つの頑健性基準を全て満たしました。

**ただし、この判定は5d(Primary Horizon)における結果です。**
第7-8・17・19-20節で詳述したとおり、10d・15dではBeta Residual・
TOPIX-relative Targetの優位性が消失(符号反転)します。**唯一
全4 Horizonで一貫して正かつ有意なのはSector-relative Target**です。
「MLで市場中立的な株式選別ができる」という単純な結論ではなく、
「特定のHorizon(5d・20d)・特定のTarget定義(Beta Residual・
Sector-relative)において、頑健な市場中立エッジの証拠が見つかった」
という、より正確で限定的な結論として報告します。

## 26. Limitations

- **本判定は依然として同一の2022-2026年サンプル内での検証です**
  (指示書section 12の明示的な注記どおり、「Independent OOS」では
  なく「Independent Target/Model Specification Validation on the
  same historical OOS framework」と表記します)。将来の未知データ
  での再現性は未実証です。
- **10d・15dでのspread符号反転**(第8・17節)は、STOCK_SELECTION_
  EDGE判定の頑健性に対する重要な留保です。全Horizonで一貫した
  エッジではありません。
- Beta推定(60日ローリング、`MAX_PLAUSIBLE_BETA=5.0`)は本Phase・
  V3-4共通の分析専用の量であり、V3のFeature Registryには一切
  追加されていません。
- Matched Controlの`scale`分類は、既知の制約(JPXの現在日スナップ
  ショットを2022-2026年全期間に投影)を引き継いでいます。
- `target_vol_adjusted_5d`列は、意図的に未フィルタの比率Targetで
  あり、極端値の影響を受けやすいことに注意してください(第18節)。
- Sharpe/MaxDD/Annualized Returnは、V3-3/V3-4から引き継いだ「重複
  トレード」設計上のバイアスを含みます。

## 27. Next Phase Gate

本Phaseの結果を見て、V3のFeature/Target/Model/Hyperparameter/Score/
Threshold/Top-N条件のいずれも変更していません。Hyperparameter
tuning・Feature Selection・Model Ensemble・V1/V2統合・実運用・UI
実装のいずれにも進みません。**指示書の最終ルールどおり、本Phase
完了をもって停止します**。

STOCK_SELECTION_EDGEという結果は、V3プロジェクトにとって最も有望な
兆候ですが、10d/15dでの符号反転という重要な留保付きです。次の
Phase(V3-6以降)を開始する場合は、本報告書のレビュー後、明示的な
指示を受けてから、以下のような**別途事前登録された**検証(例:
Beta Residual/Sector-relative Targetを用いたRisk-adjusted Ranking
Engineの正式な構築、Forward Test T0以降のPost-T0データでの真の
独立OOS検証)を検討することになります — ただし、それは本Phaseの
結果を見た上でのモデル改善には該当しない、別Phaseとしての新たな
事前登録が必要です(指示書section 36の明示的な境界線)。

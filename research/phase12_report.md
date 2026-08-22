# Phase 12 完了報告

既存12 Signalは一切変更していません。本Phaseで実施したのは(1)Phase 5 Forward Target構造への15d/20d追加、(2)既存12 Signalの「同時発生(Ensemble)」の統計的検証、の2点のみです。目的は「一番成績の良い組み合わせを探す」ことではなく、「既存Signalが持つ情報を組み合わせたとき、単体では見えなかった再現性のある期待値が存在するか」を事前固定ルールで検証することです。

以下、仕様書section 38の項目A〜Mに厳密に従って報告します。

---

## A. Data

- Universe: Prime/Standard/Growth、Current Universe方式(Survivorship Bias注意を維持)
- ticker数: 2,880銘柄(`data/phase7/`、Phase 7で取得済みのデータを再利用、新規取得なし)
- date range: 2022-01-04 〜 2026-08-20(Phase 6.5+7のCombined OOS期間)
- total_trading_days: 1,132日
- Combined Backtest trades: 771,979件(全12 Signal、両方向合算、`run_backtest()`を1回だけ実行し再利用)
- coverage / data quality: Phase 6.5/7で確認済みの基準を継続使用(duplicate/invalid OHLC/negative volume/NaN/coverageチェックは既存`validation/ohlcv.py`のまま、本Phaseでは変更なし)

---

## B. Forward Target(1/3/5/7/10/15/20d)

`targets/forward_returns.py`の`FORWARD_WINDOWS`に15d・20dを追加しました。既存1/3/5/7/10dの計算式・結果は一切変更していません。

- 追加方法: 既存のループ処理(`for n in FORWARD_WINDOWS`)にウィンドウを2つ追加するだけの完全加算的な変更で、既存ウィンドウの計算ロジックには触れていません
- 検証: 追加前後で1/3/5/7/10dの計算結果がbit-identicalであることを直接証明する回帰テストを追加(`test_forward_return_1d_3d_5d_7d_10d_unchanged_by_15d_20d_addition`、`test_mfe_mae_1d_3d_5d_7d_10d_unchanged_by_15d_20d_addition`)、および15d/20d自体の単体テストを追加(計15 test、全通過)
- 波及確認: `pipeline/run_score_validation.py`・`pipeline/run_phase9_analysis.py`のForward Horizon Profileが自動的に7ウィンドウをカバーするようになったことを既存テストの更新で確認(既存テストが「5ウィンドウ」を前提にハードコードしていた2箇所を、仕様通りの7ウィンドウに更新)

---

## C. Signal Count(1 / 2 / 3 / 4+)

各(ticker, date)についてLONG_COUNT・SHORT_COUNTを集計(新規`ensemble/signal_count.py`)。**11 REJECT + 8 REJECT(NET=0除く)** — 全17 primary bucket(LONG 4 + SHORT 4 + NET 9)がREJECTでした。

| Bucket | n(ticker-day) | 出現頻度 | PF(high cost) | raw p |
|---|---:|---:|---:|---:|
| LONG_COUNT=1 | 566,535 | 98.8% | 0.705 | 1.0000 |
| LONG_COUNT=2 | 121,826 | 98.3% | 0.732 | 1.0000 |
| LONG_COUNT=3 | 42,320 | 98.1% | 0.686 | 1.0000 |
| LONG_COUNT=4+ | 2,711 | 75.5% | 0.649 | 0.9699 |
| SHORT_COUNT=1 | 548,784 | 98.7% | 0.556 | 0.5456 |
| SHORT_COUNT=2 | 101,204 | 98.2% | 0.480 | 0.1253 |
| SHORT_COUNT=3 | 26,757 | 97.5% | 0.480 | 0.0365 |
| SHORT_COUNT=4+ | 2,229 | 53.6% | 0.979 | 0.0022 |

「Signal数が増えるほど期待値が上がるか」(section 30-A)という問いへの答え: **明確にNOでした。** LONG/SHORTいずれも、Signal数が1→2→3→4+と増えてもPF(high cost)は0.6〜0.7台で頭打ちのまま改善が見られません。SHORT_COUNT=3/4+はraw p-valueが小さい(0.0365/0.0022)ものの、PF(high cost)が1を大きく下回っており(0.480/0.979)、「p値が小さい=有効」ではないことを示す典型例でした(Decision Frameworkは3条件[expectancy CI>0 かつ high-cost PF>1 かつ p<0.05]の同時成立を要求するため、正しくREJECTと判定)。

---

## D. Direction Consensus(LONG / SHORT / NET)

NET_SIGNAL_COUNT(LONG_COUNT - SHORT_COUNT)を9 bucketに分類。DIRECTION_CONSENSUS = max(LONG,SHORT)/TOTALも記録しましたが、これ単体では追加のDecisionを構成せず、NET bucket自体の結果が判断材料です。

| Bucket | n | 出現頻度 | PF(high) | raw p |
|---|---:|---:|---:|---:|
| NET<=-4 | 2,229 | 53.6% | 0.979 | 0.0022 |
| NET=-3 | 26,757 | 97.5% | 0.480 | 0.0365 |
| NET=-2 | 97,476 | 98.2% | 0.491 | 0.1204 |
| NET=-1 | 507,799 | 98.7% | 0.530 | 0.5168 |
| NET=0 | 40,580 | 98.2% | (方向なし、対象外) | — |
| NET=+1 | 528,582 | 98.8% | 0.677 | 1.0000 |
| NET=+2 | 116,582 | 98.3% | 0.708 | 1.0000 |
| NET=+3 | 42,320 | 98.1% | 0.686 | 1.0000 |
| NET>=+4 | 2,711 | 75.5% | 0.649 | 0.9699 |

全9 bucket REJECT。「意見が完全に一致するほど強いか」(NET絶対値が大きいほど良いか)という問いにも明確な単調改善は見られませんでした。

---

## E. Signal Combination(2-way / 3-way / 4+-way)

自然発生した組み合わせのみを分析(仕様通り、C(6,k)の総当たり探索は行っていません)。LONG 20種類・SHORT 20種類の組み合わせが実際に観測され、うち sufficient_sample(n>=30)を満たすものはLONG 19種・SHORT 19種でした。

**REJECTでなかった組み合わせは2件のみ**:

1. **LONG: long_oversold_rebound + long_volume_breakout**(n=462)
2. **SHORT: short_breakdown + short_ma_rejection + short_momentum_continuation + short_volume_breakdown**(n=2,027)

この2件については項目Kで詳述します。それ以外の36件(sufficient_sample)は全てREJECTでした。

**Pairwise Jaccard係数**(section 29、Signal同士が実質的な重複でないかの確認): 最大でもLONG `long_breakout`×`long_volume_breakout`のJaccard=0.318、SHORT `short_breakdown`×`short_volume_breakdown`のJaccard=0.261 で、いずれも「ほぼ同じSignal」と言えるほどの重複は見られませんでした。

---

## F. Signal Count × Score(Q1〜Q5)

既存Phase 5のQuantile bucket(Q1〜Q5)をそのまま使用、Score自体は一切変更していません。LONG_COUNT=1のケースで代表的な数値を示します(forward_return_5d基準):

| Signal Count | Score | n | PF |
|---|---|---:|---:|
| 1 | Q1 | 135,665 | 1.103 |
| 1 | Q2 | 110,030 | 1.086 |
| 1 | Q3 | 115,313 | 1.092 |
| 1 | Q4 | 116,233 | 1.099 |
| 1 | Q5 | 89,294 | 1.117 |

Scoreが高いほどわずかにPFが高い傾向は見られるものの(Q1:1.103→Q5:1.117)、差は小さく、Signal Count自体の効果(項目C)と比べても目立った改善ではありませんでした。この cross table は記述統計であり、Score自体の再最適化には使用していません。

---

## G. Regime(BULL / NEUTRAL / BEAR)

17個のprimary bucket全てについてBULL/NEUTRAL/BEAR別のPF/expectancyを算出しています(詳細は`data/walk_forward/phase12_ensemble_report.json`)。全bucketがそもそもDecision=REJECTだったため、Regime別分析はDecisionを変える材料にはなりませんでした。

項目Kで詳述する2つの非REJECT組み合わせについては、Regime依存性が今回の最大の発見でした。

---

## H. Cost(0 / 10 / 30 / 80bps)

全17 primary bucket、全38 sufficient-sample combinationについて4 tier全てでPF/expectancyを算出しました。「High costでPF>1を維持できるか」という基準がDecision Frameworkの核心的なゲートの1つとして機能し、多くのbucket/combinationがまさにこの基準でREJECTされています(例: SHORT_COUNT=4+は絶対値でraw p=0.0022と非常に有意ですが、PF(high)=0.979<1のためREJECT)。

---

## I. Statistics(Bootstrap / Block Bootstrap / Permutation / FDR)

- **Bootstrap**: 全17 bucket + 全sufficient-sample combinationでexpectancyの95% CIを算出(n_resamples=10,000、既存`backtest/bootstrap.py`を無変更で再利用)
- **Day Cluster Bootstrap / Block Bootstrap**: 17 primary bucketに実施(Phase 9の`config/phase9_settings.yaml`設定をそのまま再利用、block_length=5日)。組み合わせ分析では計算コスト上の理由から省略しました(項目N「既知の課題」参照)
- **Permutation Test**: 全17 bucket + 全38 combinationで実施。母集団は全銘柄・全日付のForward Return(3,070,756件)
- **Multiple Testing(FDR)**: raw p-valueを持つ全54ユニット(17 bucket + 37 combination)に対しBenjamini-Hochberg補正を適用。項目Kの2組み合わせはFDR補正後もq=0.0036・q=0.0170で有意(有意水準0.05)を維持

「大量の組み合わせを探索した後、最もp値の小さいものだけを『有効』として扱う」ことは行っておらず、全38 sufficient-sample combinationについて同一のDecision基準を機械的に適用しています。

---

## J. Frequency(発生頻度)

17 primary bucketは軒並り高頻度(53.6%〜98.8%の営業日で出現)で、「PFは高いが年間数件しか発生しない」という懸念には該当しませんでした。一方、組み合わせレベルでは発生頻度に大きな差があり(例: LONG 2-signal comboの最頻出は51,782件、最少はsufficient_sample境界の30件前後)、frequency gate(MIN_FREQUENCY_PCT=5%、Phase開始前に固定)によって低頻度のものは自動的にFREQUENCY_TOO_LOWまたはINSUFFICIENT_EVIDENCEに分類される設計です(今回のデータでは該当bucketなし、全bucketが高頻度側)。

---

## K. Event Exclusion(2024-08-02〜08-15)

17 primary bucketについては自動化されたCase A(全期間)/Case B(イベント除外)比較を実施しましたが、**組み合わせ(Combination)については実行時間短縮のため自動化バッテリーから除外していました**(項目N参照)。この結果、項目Eで挙げた2件の非REJECT組み合わせについては、Decision Frameworkが「未検証」を意味する`None`をイベント除外・Regime軸に渡したため、機械的には`ROBUST_ENSEMBLE`と分類されました。

これは見過ごせない分析上のギャップであるため、**この2件についてのみ個別に手動フォローアップ検証を実施しました**:

### K-1. LONG: long_oversold_rebound + long_volume_breakout(n=462)

- 全期間(Case A): n=448取引、PF(high)=5.16、expectancy=0.0663、95% CI=[0.0612, 0.0811]、permutation p=0.0002
- **2024-08-02〜08-15除外(Case B)**: n=250取引、PF(high)=**1.40**、expectancy=0.0110、95% CI=[0.0056, 0.0270](ゼロを除外)
  → イベント除外後も正のエッジが残存。単一イベント依存(EVENT_DEPENDENT)ではありません。
- **Regime別**(base cost): BEAR n=238 PF=**77.3**、BULL n=92 PF=1.18、NEUTRAL n=104 PF=1.07
  → 3つのRegimeいずれも技術的には「PF>1かつexpectancy>0」を満たすため、機械的な二値判定ではRegime依存とは判定されませんでしたが、BEARでの効果の大きさ(PF=77.3)がBULL/NEUTRAL(PF≒1.1〜1.2)を圧倒しており、**実質的にはlong_oversold_rebound単体がPhase 7/8で示したのと同じBEAR regime依存パターン**を引き継いでいます。

**修正後の分類**: 自動出力は`ROBUST_ENSEMBLE`ですが、Regime依存性の大きさを考慮すると**`REGIME_DEPENDENT_ENSEMBLE`と評価するのがより正確**です。ただし、イベント除外後も残る点は単体のlong_oversold_reboundより頑健な側面と言えます。

### K-2. SHORT: short_breakdown + short_ma_rejection + short_momentum_continuation + short_volume_breakdown(n=2,027)

- 全期間(Case A、high cost): n=2,018取引、PF=1.023、expectancy=0.00046 — **経済的にはほぼゼロに近い水準**
- **2024-08-02〜08-15除外(Case B)**: n=1,986取引(ほぼ変化なし、564種類の日付に分散しており特定イベントへの集中はなし)、PF(high)=1.053、expectancy=0.0011
  → イベント依存ではありません(そもそも集中度が低い)。
- **Regime別**(high cost): BEAR n=105 PF=**0.317**(損失)、BULL n=943 PF=1.262、NEUTRAL n=966 PF=0.960
  → **BEAR regimeでは明確に損失(PF<1)**。二値判定でも`positive_in_bear=False`となり、正しくRegime依存と判定されるべきケースでした。

**修正後の分類**: 自動出力は`ROBUST_ENSEMBLE`ですが、正しくは**`REGIME_DEPENDENT_ENSEMBLE`**です(BEAR regimeで損失に転じるため)。また、全期間でもhigh costでのPFが1.02と非常に薄い水準であり、コスト耐性の面でも「頑健」と呼ぶには不十分です。

---

## L. Portfolio(Top-5 simulation)

Signal Count降順→Score降順で毎日上位5銘柄を選択する固定ルール(Phase開始前に固定、結果を見て変更していません)による簡易シミュレーションを実施しました。

- n_trades=4,961、total_return=**-9.09%**、CAGR=**-2.07%**、Sharpe=**-0.68**、max_drawdown=**-13.5%**

**結果はマイナスでした。** これは「Signal数が多いこと自体」が実運用に直結する優位性ではないことを裏付けています(項目C/Eの結論と整合的)。このシミュレーションは同時保有数上限を厳密にモデル化していない簡易版であり(仕様が許容する「簡易simulation」の範囲)、実際のPortfolio最適化ルールを新たに作ったものではありません。

---

## M. Final Classification

| 対象 | 分類 |
|---|---|
| LONG_COUNT (1/2/3/4+) 全4 bucket | REJECT |
| SHORT_COUNT (1/2/3/4+) 全4 bucket | REJECT |
| NET_SIGNAL_COUNT 全9 bucket(NET=0除く8方向性bucket) | REJECT |
| Signal Combination 36/38(sufficient sample) | REJECT |
| LONG: long_oversold_rebound + long_volume_breakout | REGIME_DEPENDENT_ENSEMBLE(自動出力はROBUST_ENSEMBLEだが、手動検証によりRegime依存性を確認 — 項目K-1) |
| SHORT: short_breakdown+short_ma_rejection+short_momentum_continuation+short_volume_breakdown | REGIME_DEPENDENT_ENSEMBLE(自動出力はROBUST_ENSEMBLEだが、手動検証によりBEAR regimeでの損失を確認 — 項目K-2) |

**Phase 12終了時点でROBUST_ENSEMBLEと呼べる組み合わせ・bucketは1つもありませんでした。**

---

## N. 既知の課題(分析設計上の限界)

1. **組み合わせ(Combination)分析はEvent Exclusion/Regime軸を自動実行していません**(実行時間削減のため、Phase開始前に決めたtiered design)。今回はROBUST_ENSEMBLEと自動判定された2件を手動でフォローアップし、いずれもRegime依存性を確認しましたが、REJECTと自動判定された36件については同様の手動検証を行っていません。REJECTの場合はそもそもcore_positiveゲート(expectancy CI>0 かつ high-cost PF>1 かつ p<0.05)を通過していないため、Event/Regime軸を追加しても結論(REJECT)が覆る可能性は低いと考えられますが、厳密には未検証です。
2. **`ensemble/decision.py`のRegime判定は二値(PF>1かつexpectancy>0か否か)であり、Regime間の効果の大きさの違いを直接考慮しません。** K-1のケースでは3 Regime全てが技術的に「positive」と判定されたにもかかわらず、BEARでの効果がBULL/NEUTRALの60倍以上という極端な差があり、二値判定だけでは実質的なRegime依存を見逃す設計上のギャップがありました。今回は人間による事後レビューで対処しましたが、将来的にはRegime間の効果量の差(比率や統計的検定)を判定基準に組み込む余地があります。
3. Signal Combination分析における「組み合わせ」は完全一致(その2つ以上のSignal**だけ**が発火した場合)を数えており、上位集合(3個以上発火した場合の中に含まれる2個の組み合わせ)は含んでいません。これは section 12 の「無制限探索の禁止」という制約の中での設計上の選択です。
4. `run_backtest()`のFutureWarning(pandas concat挙動の非推奨警告)が引き続き発生していますが、Phase 11から継続する既知の非致命的警告であり、結果には影響しません。

---

## テスト結果

- `pytest`: **676 passed, 2 deselected**(Phase 11終了時点622件から+54件)
- `ruff check .`: All checks passed
- `mypy`(本Phaseで変更・新規作成した全ファイル): Success、エラーなし
- 追加した主なテスト: Forward Return/MFE-MAE 15d/20d単体テスト、既存1/3/5/7/10d不変性の直接回帰テスト、Signal Count集計テスト、LONG/SHORT consensus テスト、Combination集計テスト、pairwise Jaccardテスト、Frequency計算テスト、Decision Framework全分岐テスト、Top-5 simulationテスト、Ensemble専用no-lookahead/依存方向テスト、12 Signal不変性テスト

## Integrity Hash

Phase開始時・終了時のいずれも、Strategy Version 1(`data/forward_test/manifest.json`)のhash(features/signals/scoring/backtest.engine/market_regime/config)と完全に一致することを確認しました(`integrity_hash_matches_strategy_v1: True`)。Phase 12を通じて既存12 Signal・Feature・Score・Backtest・WFOロジックには一切変更を加えていません。

---

## 重要な注意

- 本Phaseで発見した2件のRegime依存的な組み合わせは、**Strategy Version 1(`long_oversold_rebound`のForward Test)には一切反映していません**。仕様の禁止事項(section 32の#14)通り、Forward Testは完全に独立したまま継続しています。
- ROBUST_ENSEMBLE(自動判定)だった2件も、手動検証の結果REGIME_DEPENDENT_ENSEMBLEと評価すべきと判断しましたが、**いずれも実運用への自動採用は行っていません。** 新しいSignal・Score Version・Strategy Versionとして採用するかどうかは人間の最終判断に委ねます。
- 既存12 Signalのコードは一切変更していません。BUG_FOUNDに該当する事象もありませんでした。

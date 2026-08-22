# Phase V2-1 完了報告: Swing Candidate Ranking Engine

Strategy Version 1(既存12 Signal・`long_oversold_rebound`・Score・Backtest・
Walk Forward Validation・Phase 10 Forward Test Engine)は**完全凍結**の
まま、一切変更していません。本Phaseは、V1とは独立した新しい研究系統
`v2/`パッケージの初期実装です。**利益が出る・有効な戦略であるとは
結論付けません** — これはResearch Ranking Engineの初期実装であり、
V1が既に分析済みの既存データ(2022-2026)を「動作確認用のResearch/
Development Dataset」として使っただけで、V2独自の独立OOS性能評価では
ありません。

---

## 1. 実装したFeature

`v2/features_adapter.py`が`features.pipeline.compute_feature_panel()`
(V1コード、無変更)をそのまま呼び出し、6カテゴリ・全24特徴量を
Feature Engineering層として再利用しています。V1にない4つの派生特徴量
のみをV2側で追加しました(V1の`features._utils.sma`/`safe_divide`を
再利用、新しい計算式は導入していません)。

| カテゴリ | 特徴量 | 出典 |
|---|---|---|
| Momentum | return_1d/3d/5d/10d/20d/60d | V1再利用(`features/momentum.py`) |
| Momentum | close_to_sma_5/20 | V1再利用(`features/trend.py`) |
| Momentum | price_vs_ma60 | V2新規派生(`sma()`再利用) |
| Momentum | ma5_vs_ma20, ma20_vs_ma60 | V2新規派生(既存sma列の比率) |
| Trend | sma_20_slope, sma_50_slope | V1再利用(`features/trend.py`) |
| Trend | distance_from_recent_high(20d) | V1再利用(`features/pullback.py`) |
| Trend | distance_from_60d_high | V2新規派生(同じ式、window=60) |
| Volume | volume_ratio_5d/20d, volume_trend | V1再利用(`features/volume.py`) |
| Volatility | volatility_5d/10d/20d, atr_pct | V1再利用(`features/volatility.py`) |
| Relative Strength | rs_5d/20d/60d | V1再利用(`features/relative_strength.py`) |
| Pullback/Oversold | pullback_depth, consecutive_down_days, rsi_14 | V1再利用(`features/pullback.py`, `features/indicators.py`) |

**方向性(directionality)についての重要な注記**: V1の
`features/metadata.py::FeatureMeta.directionality`はモメンタム的な
解釈(例: `pullback_depth`は"bullish_low" = 浅い押し目ほど強気)を
採用しています。V2のPullback/Oversoldカテゴリは意図的にこれと**逆**
の逆張り解釈(深い押し目・連続陰線・低RSIほど高評価)を採用しました。
これはV1自身の`long_oversold_rebound`が採用条件に`rsi_14 < 30`という
逆張り条件を使っているのと同じ発想を、離散閾値ではなく連続的な
Percentile Rankとして表現したものです。VolatilityはV1では"neutral"
(意見なし)ですが、V2では「リスクバランスの良い候補」というV2自体の
目的(指示書section 0)に基づき、低ボラティリティほど高評価としました
- これはV1のタグの流用ではなく、V2独自の新しい判断です。

未実装(spec section 6が例示したが今回は採用しなかった項目): trend
persistence(明確な操作的定義がなく、過度な複雑化を避けるため見送り)。

## 2. Forward Target

`v2/targets_adapter.py`が`targets.forward_returns.compute_forward_returns()`
(V1コード、無変更)をそのまま呼び出し、`FORWARD_WINDOWS=(1,3,5,7,10,15,20)`
全7windowを保持しています。V2自体のランキング・レポートでは
5/10/15/20dの4windowに焦点を当てています(`v2/config/v2_settings.yaml::forward_windows`)。

## 3. Score構造

各カテゴリについて、メンバー特徴量それぞれをUniverse横断で日次
Percentile Rank化し(`v2/ranking/cross_sectional.py::percentile_rank_by_day`)、
カテゴリ内で単純平均してカテゴリRankを算出します
(`compute_category_ranks`)。最終的なV2 Scoreは、6つのカテゴリRankの
固定weight加重和です(`compute_v2_score`)。カテゴリRankがすべて
NaN(=Feature warmup中等)の行は、Scoreも NaN のまま除外され、他の
行と異なる少ないカテゴリ数でScoreが計算されることはありません
(補完・re正規化なし)。

## 4. Weight(V2 Initial Research Score)

`v2/config/v2_settings.yaml`に固定・記録済み。**既存データ
(Phase 6.5〜14)を見て調整したものではありません**。

| カテゴリ | Weight |
|---|---:|
| Momentum | 25% |
| Trend | 20% |
| Volume | 15% |
| Relative Strength | 20% |
| Pullback | 10% |
| Volatility | 10% |

## 5. Universe

Phase 6.5/7/12/13と同じFull Universe(`data/phase7/`)を、
`pipeline.universe_ingest.load_manifest()`(V1コード、無変更)経由で
READ ONLYで参照しています。`included_in_universe=True`のみ採用。
Prime/Standard/Growth、ETF/REIT除外はV1のUniverse構築時点の設定
そのまま(V2は独自のUniverseフィルタリングを行っていません)。
Current Universe方式によるSurvivorship Biasが存在することを明記します
(V1の既存注記と同じ扱い)。

## 6. Candidate生成方法

`v2/candidate.py::build_candidate_table()`が、1日分のUniverse横断
Score降順で(date, ticker, score, rank, score_percentile,
classification, market, category_ranks, forward_returns)を保持する
`CandidateRecord`のリストを生成します。`classify_candidate()`は
**完全に機械的**なScore percentileのみによる3分類
(CANDIDATE: 上位20%、AVOID: 下位20%、WATCH: それ以外)で、それ以外の
シグナル・判断は一切組み込んでいません。「買い」と断定するUI・
ロジックはありません。

## 7. Score Q1-Q5結果 / 8. 5/10/15/20d比較(Holding Period Analysis)

`scripts/run_v2_research_dry_run.py`をFull Universe(2,880銘柄、
2022-01-04〜2026-08-20、`data/phase7/`)に対して実行。全3,085,158行
中、Feature warmup等でtotal_scoreが算出できた行は3,030,424行
(98.2%)。`assign_quantile_buckets()`(V1コード、無変更)でQ1〜Q5に
5分位。

**データ整合性に関する重要な発見**: 生の集計では、Q2〜Q4bucketの
平均Forward Returnが数十〜数百(=数千%〜数万%)という物理的にあり
得ない値になりました。原因調査の結果、銘柄`8303`の2023年9月付近の
4行が5日Forward Returnで約1,978万%(!)という明らかなデータ異常
(株式分割の未調整、または生データフィードの誤りと推定)を含んで
いたことが判明しました。これはV1のSignal-gated分析(特定の
トリガー条件を満たす行のみを見る)では表面化しなかった可能性が
高く、V2がUniverse・全営業日を対象に横断分析することで初めて
検出できた問題です。`v2/stats.py::exclude_implausible_returns()`
として、日本株の値幅制限を踏まえた固定・非チューニングの閾値
(|Forward Return| > 500%を除外、`MAX_PLAUSIBLE_FORWARD_RETURN=5.0`)
を導入し、4 window全体で**わずか5〜30行(全体の0.0002%〜0.001%)**
を除外した結果、平均値は健全な範囲に復帰しました(除外前後の詳細は
`v2/stats.py`のコメント、および本レポート脚注参照)。この除外は
結果を見てから閾値を調整したものではなく、除外前に物理的に不可能な
値だと判断できる固定基準です。

以下は除外後の集計です。

### 5d

| Bucket | n | mean | median | win_rate | PF | std | max_loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 603,313 | 0.388% | 0.000% | 49.67% | 1.205 | 6.68% | -56.17% |
| Q2 | 603,246 | 0.312% | 0.124% | 50.81% | 1.207 | 5.41% | -62.27% |
| Q3 | 603,046 | 0.283% | 0.143% | 51.20% | 1.197 | 5.16% | -67.26% |
| Q4 | 603,167 | 0.282% | 0.159% | 51.44% | 1.192 | 5.23% | -68.00% |
| Q5 | 603,262 | 0.279% | 0.115% | 50.74% | 1.152 | 6.51% | -62.83% |

Q5-Q1 spread: **-0.109%**(除外後、n_resolved=3,016,034 / 除外5行)

### 10d

| Bucket | n | mean | median | win_rate | PF |
|---|---:|---:|---:|---:|---:|
| Q1 | 600,582 | 0.678% | 0.000% | 49.83% | 1.257 |
| Q2 | 600,222 | 0.585% | 0.200% | 51.22% | 1.278 |
| Q3 | 600,030 | 0.560% | 0.247% | 51.83% | 1.281 |
| Q4 | 600,316 | 0.568% | 0.278% | 52.15% | 1.281 |
| Q5 | 600,504 | 0.598% | 0.248% | 51.60% | 1.245 |

Q5-Q1 spread: **-0.079%**(n_resolved=3,001,654 / 除外10行)

### 15d

| Bucket | n | mean | median | win_rate | PF |
|---|---:|---:|---:|---:|---:|
| Q1 | 597,863 | 0.948% | 0.000% | 49.91% | 1.296 |
| Q2 | 597,372 | 0.838% | 0.253% | 51.36% | 1.329 |
| Q3 | 596,924 | 0.837% | 0.346% | 52.28% | 1.350 |
| Q4 | 597,277 | 0.863% | 0.390% | 52.60% | 1.359 |
| Q5 | 597,831 | 0.906% | 0.381% | 52.16% | 1.314 |

Q5-Q1 spread: **-0.041%**(n_resolved=2,987,267 / 除外22行)

### 20d

| Bucket | n | mean | median | win_rate | PF |
|---|---:|---:|---:|---:|---:|
| Q1 | 594,809 | 1.162% | 0.057% | 50.05% | 1.318 |
| Q2 | 594,794 | 1.121% | 0.344% | 51.74% | 1.389 |
| Q3 | 594,144 | 1.152% | 0.463% | 52.72% | 1.428 |
| Q4 | 594,351 | 1.167% | 0.522% | 53.10% | 1.433 |
| Q5 | 594,789 | 1.188% | 0.474% | 52.34% | 1.364 |

Q5-Q1 spread: **+0.027%**(n_resolved=2,972,887 / 除外30行)

**観察(統計的検定は未実施 - 記述統計のみ、指示書section 25の通り
「有効である」と結論付けません)**:

- Q5-Q1 spreadは4 window全てで**ごくわずか**(-0.11%〜+0.03%)であり、
  一貫した単調増加のパターンは見られません。むしろ5d/10d/15dでは
  Q5がQ1よりわずかに低い(逆行)結果でした。
- Win Rateは Q1→Q4 にかけて緩やかに上昇する傾向(例: 20dで50.1%→
  53.1%)が見られますが、Q5では毎windowともQ4より低下しており、
  単純な「Scoreが高いほど良い」という単調な関係は確認できません。
- Profit Factorも同様のパターン(Q1〜Q4で緩やかに上昇、Q5で低下)。
- Holding Period(5d→20d)が長くなるほどmean/median/PFはいずれも
  緩やかに改善する傾向が見られますが、これはScore Bucket間の差では
  なくWindow間の差であり、単に長期保有ほど市場全体の期待リターンが
  積み上がる効果である可能性が高く、V2 Score自体の効果とは分離
  できていません。
- これらはV2 Initial Research Score(固定weight、後付け最適化なし)
  の**初期実装バージョンでの記述統計**であり、統計的有意性検定
  (Bootstrap/Permutation等)は本Phaseでは未実施です。「V2 Scoreは
  現時点でこのRD Datasetでは強い予測力を示していない」という記述的
  事実を報告するに留め、Score構造・weightを本Phase内で調整すること
  は行っていません(指示書section 25の禁止事項)。

## 9. Leakage Test

`tests/test_v2_leakage.py`:

1. **静的(AST)依存方向検証**: `features/signals/scoring/backtest/targets/ensemble/forward_test/pipeline`
   配下のV1モジュールが`v2`を一切importしていないことを確認
   (`test_v1_packages_never_import_v2`)。逆方向(V2がV1からimportして
   いること自体)も、reuseが実際に行われていることのsanity checkとして
   確認しています。
2. **Future Shock Test**: 1銘柄の最終日の株価を5倍に変更した合成データを
   用意し、変更前と変更後でそれ以前の全日付・全銘柄の
   Feature Rank/Category Rank/Total Scoreが完全一致することを確認
   (`test_future_shock_never_changes_earlier_dates`)。Cross-sectional
   Rankingが日ごとに独立している(`groupby(date_col)`)ため、ある銘柄の
   未来のショックが同じ銘柄はもちろん、他のどの銘柄の過去日付にも
   一切影響しないことを直接検証しています。

## 10. Determinism Test

`tests/test_v2_determinism.py`: 同一の合成Universe・同一Configで
`run_v2_ranking()`を2回実行し、結果(全列)が完全一致することを
`pandas.testing.assert_frame_equal`で確認。Candidate Tableについても
同様に2回実行して同一順序・同一値であることを確認しています。V2は
MLを使用しないルールベース・統計ベースの実装であり、未シード化された
乱数は一切使用していません。

## 11. Test件数

V2専用テスト **64件**、全てPASS(`tests/test_v2_*.py`、11ファイル)。
内訳: Feature Adapter 7件、Target Adapter 3件、Cross-sectional
Ranking 8件、Score 6件、Candidate 8件、Determinism 2件、Manifest 4件、
Stats 11件(データ整合性の`exclude_implausible_returns()`用3件を
本Phase中に追加)、Config 4件、Pipeline統合 6件、Leakage 5件。
プロジェクト全体のregression floor(V1既存分含む): **775 passed** /
2 deselected(V2追加64件を含む、V1側の回帰なし)。ruff/mypy両方clean
(`v2/`配下12ファイル)。

## 12. V1への影響がないこと

`git status`で確認: 本Phaseで変更されたのは`.gitignore`
(V2出力ディレクトリの除外ルール追加のみ)と`README.md`
(V2セクション追加のみ)の2ファイルで、いずれも**追記のみ**、既存V1
関連の記述は一切変更していません。それ以外はすべて新規ファイル
(`v2/`パッケージ、`tests/test_v2_*.py`、`scripts/run_v2_*.py`、本
レポート)です。V1のSignal/Score/Backtest/Walk Forward
Validation/Forward Test Engine/config/ログ/Forward Test stateは
一切変更していません。Strategy Hash・V1のconfig_hashも本Phase中
変更していません(V2はこれらを一切書き換える処理を持ちません)。

## 13. 今回まだ評価していない事項

- **V1データへの影響未確認**: 本Phaseでticker `8303`の2023年9月付近に
  明らかなデータ異常(未調整の株式分割等と推定)を発見しました。V2は
  この4行を除外して対応しましたが、**V1側のデータ(`data/phase7/`)
  そのものは一切変更していません**。この異常がV1の既存Phase(6.5〜14)
  の分析結果(特に`long_oversold_rebound`や他11 Signalのbacktest)に
  影響しているかどうかは未確認です - V1はSignal-gatedな分析
  (特定条件を満たす行のみ)であるため、この特定の異常データがどの
  Signalの条件にも該当していなければ影響はない可能性が高いですが、
  確認はしていません。V1側で確認・対応する場合も、本Phaseの範囲
  外であり、ユーザーの指示なしに着手しません。
- **trend persistence**特徴量(明確な操作的定義を持たせられなかった
  ため見送り)。
- V1の`compute_feature_panel()`を毎回フルスキャンで再計算する経路
  (`build_v2_feature_panel()`)は用意していますが、実際のFull
  Universe実行では未来の新規日付に対応するためのfresh computeパス
  ではなく、V1の既存キャッシュ再利用パス(`add_v2_derived_features()`)
  のみを使っています。live/最新日付に対する新規fetch・計算の
  オーケストレーションは未実装です。
- BUY/WATCH/AVOID分類は現時点でScore percentileのみの機械的な
  3分類に留まっており、他のリスク指標・出来高最低ラインなどは
  一切組み込んでいません。
- Bootstrap等の統計的検定は指示書section 14の通り今回必須ではなく、
  未実装です(Q1-Q5の記述統計のみ)。
- V2独自の独立Forward/OOS期間はまだ確保されていません(本Phaseの
  Q1-Q5結果は既存の2022-2026 Research/Development Datasetに対する
  ものであり、Phase 14のOOS独立性の議論と同じ制約を受けます)。
- Market Context(TOPIX return等)はFeature列として保存しています
  (`rs_5d/20d/60d`がTOPIX Proxy相対の値を含む)が、指示書section 7が
  想定する「Market Context × Candidate Ranking」という独立検証は
  未実装です。Phase 13の「BEAR/急落局面でlong_oversold_reboundが
  強い」という知見はV2 Scoreに一切組み込んでいません。

## 14. 次Phase候補

- V2独自の独立Forward/OOS期間の確保(Phase 14と同様の制約 - Forward
  Testが十分な期間蓄積されるまで待つ必要がある)。
- Market Context × Candidate Rankingの独立検証(Phase 13知見の
  V2側での再検証、後付け最適化なしで)。
- BUY/WATCH/AVOID分類の精緻化(ただし引き続き「買い」の断定は行わない)。
- Bootstrap/Permutation等の統計的検定の追加。
- 新規日付に対するfresh computeパスの整備(live運用を見据える場合)。

いずれもユーザーの明示的な指示があるまで着手しません。本Phaseは
ここで停止します。V1のいかなる変更(Signal/Score/Backtest/Strategy
Version 2作成等)にも進みません。

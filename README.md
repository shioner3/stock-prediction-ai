# Swing Scanner

東証上場個別株の日足データを用いたスイングトレード候補検出・検証ツール。

**これは完全自動売買システムではありません。最終的な売買判断は人間が行います。**
投資助言ではありません。

設計方針の詳細は設計ドキュメント(v0.2)を参照してください。要点:

- Signal(セットアップ発生の有無)とScore(品質評価)を分離する
- 実装したSignal/Scoreは「有効である」と仮定しない。HYPOTHESIS → IMPLEMENT →
  BACKTEST → OUT-OF-SAMPLE TEST → EVALUATE → ACCEPT/REJECT のプロセスを経て
  初めて採用する
- 未来の情報を現在のシグナル判定に使用しない(No-lookahead)ことを最優先とする

## 現在の実装状況: Phase 13(long_oversold_rebound Conditional Analysis)

Phase 1(データ取得)・Phase 2(テクニカル特徴量)・Phase 3(Relative
Strength)・Phase 4(Signal Architecture + Backtest Engine)・Phase 5
(Scoring + Score Validation)に加え、Phase 6で**Walk Forward検証**
(TRAIN→VALIDATION→OOSの時系列分割)、**Transaction Cost感応度**、
**Bootstrap信頼区間**、**Permutation Test**、**Market Regime別分析**、
そして初めて許可される**Signal/Score採否候補判定**
(ACCEPT_CANDIDATE/REJECT/INSUFFICIENT_EVIDENCE)を実装しました。
Phase 6.5では同一ロジックのままUniverseをJPX公式データの
Prime+Standard+Growth全銘柄(Final Universe 2,755銘柄)に拡張し、
12 Signal中11 REJECT/1 INSUFFICIENT_EVIDENCE/0 ACCEPT_CANDIDATEという
結果を得ました(詳細は下記「Phase 6.5」節、全文は
[research/phase6_5_full_universe_report.md](research/phase6_5_full_universe_report.md))。
Phase 7では2024-07-01〜2026-08-20という完全独立な未来期間で同一ロジックを
再検証し、12 Signal中11 REJECT/0 INSUFFICIENT_EVIDENCE/1 ACCEPT_CANDIDATE
(`long_oversold_rebound`、ただしBEAR regime依存が強い)という結果を
得ました(詳細は下記「Phase 7」節、全文は
[research/phase7_final_report.md](research/phase7_final_report.md))。
Phase 8では`long_oversold_rebound`のBEAR regime依存性をさらに検証し、
Combined OOS(Phase 6.5+7通し)ではACCEPT_CANDIDATEだったものの、その優位性
の大半(累積リターンの71.6%)が2024年8月の1回の市場イベントに集中している
ことを発見しました(詳細は下記「Phase 8」節、全文は
[research/phase8_report.md](research/phase8_report.md))。
Phase 9ではさらに、Day Cluster/Block Bootstrap・Timing Placebo Sweep・
Leave-One-Episode/Year-Out等で有効性が崩れる条件を積極的に探し、最終的に
`Primary: EVENT_DEPENDENT` / `Secondary caveat: REGIME_DEPENDENT`という
分類に至りました(詳細は下記「Phase 9」節、全文は
[research/phase9_report.md](research/phase9_report.md))。
Phase 10では過去データの分析から離れ、Signal・Score・Backtestを完全
凍結した上で未来の市場に対するPaper Trading(Forward Test)基盤を
構築し、T0=2026-08-20時点でEngine Readyを確認しました(詳細は下記
「Phase 10」節、全文は
[research/phase10_report.md](research/phase10_report.md))。実際の
Forward Test結果が蓄積するまでは戦略の有効性について結論を出しません。
Phase 11ではForward Testの自動日次実行(SAFE_ABORT・Open Position・
Daily Performance Log)を実装する一方、`long_oversold_rebound`以外の
残り11 SignalをPhase 6.5-9と同等以上の厳密さで独立検証し、11 Signal
全てREJECTという結果を得ました(両Trackは完全に独立、詳細は下記
「Phase 11」節、全文は
[research/phase11_report.md](research/phase11_report.md))。
Phase 12ではForward Target評価期間を15d/20dに拡張した上で、既存
12 SignalのEnsemble(同時発生)を検証し、全17 primary bucketと
sufficient-sampleな組み合わせ38件中36件がREJECT、残り2件も手動
検証でRegime依存性が判明し、ROBUST_ENSEMBLEと呼べる結果は1つも
ありませんでした(詳細は下記「Phase 12」節、全文は
[research/phase12_report.md](research/phase12_report.md))。
Phase 11AではForward TestをGitHub Actionsで完全自動化し(詳細は下記
「Phase 11A」節、全文は
[research/phase11a_report.md](research/phase11a_report.md))、
Phase 13では`long_oversold_rebound`の過去発生行を条件別に事後分析し、
BEAR regime・市場大幅下落局面でPFが強まる一方Score水準にはほぼ
依存しないという探索的仮説を得ました(いずれも未検証、詳細は下記
「Phase 13」節、全文は
[research/phase13_report.md](research/phase13_report.md))。
Streamlit UIは未実装です(Phase 14以降)。

**最重要ルール**: Phase 6ではOOS結果を見た後にSignal条件・Score重み・
閾値・HOLD_DAYS・Entry/Exit・Backtest条件を一切変更していません。
OOS期間は実行前にconfig/READMEへ固定し(下記「OOS期間の固定」)、
決定分類(ACCEPT_CANDIDATE等)は`backtest/decision.py`の固定基準による
機械的な「候補」であり、自動採用・自動デプロイは一切行いません。
最終判断は人間が行います。

Phase 2〜6を通して最優先したのは **NO LOOKAHEAD** であることの構造的
保証と、それを検証する自動テストです。詳細は下記「Feature定義」
「Relative Strength」「Signal Architecture」「Backtest仕様」
「Score Architecture」「Score Validation」「Walk Forward構造」
「Statistical Validation」「Signal Decision」「No-lookahead対策」を
参照してください。

**重要**: `1306.T`(TOPIX Proxy)は野村アセットのTOPIX連動ETFであり、
**TOPIX指数そのものではありません**。コード・ログ・メタデータの
どこでも「TOPIX Proxy (1306.T)」と明記し、単に「TOPIX」と呼ばない
方針を徹底しています(詳細は「Relative Strength」節を参照)。

## Strategy Version 2 (V2): 独立Research Ranking Engine

**V1**: 上記の`long_oversold_rebound`を含む既存12 Signal・Score・
Backtest・Walk Forward Validation・Phase 10 Forward Test Engineは
**完全凍結**されたStrategy Version 1です。

**V2**: `v2/`パッケージは、V1とは完全に独立した新しい研究系統
(Phase V2-1: Swing Candidate Ranking Engine)です。目的は「5日後の
株価を一点予測するMLモデル」ではなく、「日本株全銘柄から今後
5〜20営業日のスイング候補として相対的に魅力的な銘柄をランキングする」
ルールベース・統計ベースのResearch Ranking Engineです。V2で得られた
結果によってV1を変更することはありません。全文レポート:
[research/phase_v2_1_report.md](research/phase_v2_1_report.md)

- V2はV1のFeature Engineering(`features/pipeline.py::compute_feature_panel()`)
  とForward Target(`targets/forward_returns.py`)を**そのままimportして
  再利用**し、V1コードは一切変更していません(`v2/features_adapter.py`
  ・`v2/targets_adapter.py`)。V1にない4つの派生特徴量
  (`price_vs_ma60`・`ma5_vs_ma20`・`ma20_vs_ma60`・`distance_from_60d_high`)
  のみをV1のutility関数(`features._utils.sma`等)経由で追加しています。
- Momentum/Trend/Volume/Volatility/Relative Strength/Pullbackの6カテゴリ
  それぞれについて、日次でUniverse横断のPercentile Rankを計算し
  (`v2/ranking/cross_sectional.py`)、カテゴリ内平均→固定weight
  (Momentum 25%/Trend 20%/Volume 15%/Relative Strength 20%/
  Pullback 10%/Volatility 10%)で合成した「V2 Initial Research Score」
  を算出します(`v2/ranking/score.py`)。weightは既存データを見て
  調整したものではなく、固定・記録済みです
  (`v2/config/v2_settings.yaml`)。
- Pullback/Oversoldカテゴリの方向性(深い下落・連続陰線・低RSIほど
  高評価)は、V1の`long_oversold_rebound`が`rsi_14 < 30`を採用条件と
  する発想と同じ「逆張り」の解釈です。V1のFeature Metadataの
  `directionality`タグ(モメンタム寄りの解釈)とは意図的に異なります -
  詳細はレポート参照。
- V2独自のconfig_hash/code_hash/data_hash(`v2/manifest.py`)を持ち、
  V1のStrategy Hash・config_hashとは完全に別の整合性検証系統です。
- V2-1では2022-2026年の既存データ(V1が既に分析済みのFull Universe)を
  「動作確認用のResearch/Development Dataset」として使用しました -
  これはV2の独立OOS性能ではありません。将来、V2独自の独立Forward/OOS
  期間を別途確保して初めて性能評価が可能になります。
- 実運用・自動発注・証券会社API接続・MLモデル・Streamlit UI・
  V1 Forward Testへの接続のいずれも実装していません。「買い」と
  断定するUIやロジックも一切ありません。

## セットアップ

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -e ".[dev]"
```

## Phase 1: データ取得の実行

```bash
python scripts/run_ingest.py
```

作業ディレクトリのパスに日本語などの非ASCII文字が含まれる環境では、
先に下記を実行してください(詳細は「既知の問題」3を参照)。

```bash
cp .venv/lib/site-packages/certifi/cacert.pem /tmp/cacert.pem
export CURL_CA_BUNDLE=/tmp/cacert.pem
export SSL_CERT_FILE=/tmp/cacert.pem
```

明示的なticker指定(config内のUniverse master listを使わず、動作確認用に直接指定):

```bash
python scripts/run_ingest.py --tickers 7203 6758 9984
```

## Phase 2/3: Feature計算の実行

`data/processed/` にあるOHLCVから特徴量パネル(Phase 2の
Trend/Momentum/.../Pullback + Phase 3のRelative Strength)を計算し、
`data/features/` にParquet保存します(`data/raw/`・`data/processed/`
は変更しません)。`data/processed/TOPIX.parquet`(実体は`1306.T`)が
存在すればRelative Strengthも計算され、存在しなければRS列だけが
NaNになります(他のFeatureは正常に計算されます)。

```bash
python scripts/run_build_features.py
```

明示的なticker指定:

```bash
python scripts/run_build_features.py --tickers 7203 6758
```

## Phase 4: Signal計算とBacktestの実行

`data/features/` のFeature Panelから12種類のSignalを評価し、
発生した(triggered=True)行だけを`data/signals/`にParquet保存します。

```bash
python scripts/run_build_signals.py
```

続けてBacktestを実行(`data/signals/` + `data/features/`のOHLCVを使用):

```bash
python scripts/run_backtest.py --save-trades data/backtest/trades.parquet
```

Signal毎のn/win_rate/avg_return/median_return/total_return/profit_factor/
expectancyが標準出力に表示されます。`--save-trades`は省略可能です。

## Phase 5: ScoreとScore Validationの実行

`data/signals/` の発生行に対してScore(0〜100点)を計算し、
`data/scores/`にParquet保存します。

```bash
python scripts/run_build_scores.py
```

続けてScore Validation(Bucket Analysis)を実行:

```bash
python scripts/run_score_validation.py                          # 固定幅bucket、全forward window
python scripts/run_score_validation.py --bucket-method quantile  # quantile bucket
python scripts/run_score_validation.py --forward-window 5        # 5日のみ表示
```

`(direction, signal_name, forward_window)`の組み合わせごとに、
bucket別のn/mean_return/median_return/win_rate/profit_factor/expectancy
と、単調性判定(`monotonic`)・単調性の強さの簡易指標
(`monotonicity_corr`、Pearson相関、-1〜+1)が表示されます。

## Phase 6: Walk Forward + Statistical Validationの実行

Phase 4/5の結果(`data/signals/`・`data/features/`)を**再利用**し、
新たにFeature/Signal/Scoreを計算し直すことはありません。

```bash
python scripts/run_walk_forward.py
python scripts/run_walk_forward.py --save-report data/walk_forward/report.json
```

Signalごとに、Window一覧・OOS集計指標(4つのcost tier)・Bootstrap
95%信頼区間・Permutation Test p値・window一貫性・market regime別内訳・
`ACCEPT_CANDIDATE`/`REJECT`/`INSUFFICIENT_EVIDENCE`の候補判定が表示
されます。`--save-report`でJSON全体(config_hash/data_hashを含む)を
保存できます。

## テスト

```bash
pytest                 # ネットワーク不要なユニットテストのみ(デフォルト)
pytest -m network       # 実ネットワークに接続する疎通テストも含めて実行
```

## Feature定義(Phase 2)

全Featureは `features/<category>.py` の純粋関数として実装されています
(`compute_*_features(df) -> DataFrame`)。入力は生OHLCVの
DataFrame(+ Pullbackのみ`recent_high_window`)のみで、Providerへの
アクセスや未来データの参照は一切行いません。

**命名規則**: `sma_20`, `ema_20`, `return_5d`, `close_to_sma_20`,
`atr` / `atr_pct`, `volatility_20d`, `volume_ratio_20d`,
`volume_zscore`, `rsi_14`, `macd` / `macd_signal` / `macd_hist`,
`high_20d`, `pullback_depth` のように小文字スネークケースで統一。

**warmup(最初に値が出る行)**: 各Featureには
`features/metadata.py`の`FeatureMeta.warmup_period`
(何個目の観測値から有効になるか、1始まり)が定義されており、
それより前の行は意図的にNaNのままです(0埋め・前方補完はしません)。
`tests/test_features_pipeline.py::test_feature_warmup_matches_declared_metadata`
が全Feature・全カラムについて「宣言したwarmup_periodちょうどでNaNが
終わる」ことを自動検証しています。

| カテゴリ | Feature | 定義 | warmup |
|---|---|---|---|
| Trend | `sma_N`(N=5,10,20,25,50,75,100,200) | `mean(Close[t-N+1..t])` | N |
| Trend | `ema_N`(N=5,20,50) | `EWM(Close,span=N,adjust=False)`、最初のN-1行はNaNに強制マスク | N |
| Trend | `sma_N_slope` (N=5,20,50,200) | `(SMA_N[t]-SMA_N[t-5])/SMA_N[t-5]/5` (5日間の分数変化率。傾き測定窓は常に5日固定で、MA自身の期間Nとは独立) | N+5 |
| MA Distance | `close_to_sma_N` (N=5,20,25,50,75,200) | `Close/SMA_N - 1` | N |
| Momentum | `return_Nd` (N=1,3,5,10,20,60) | `Close[t]/Close[t-N] - 1` | N+1 |
| Volatility | `atr` | True Range をWilder平滑化(期間14、初期値=最初の14件のTRの単純平均) | 15 |
| Volatility | `atr_pct` | `atr / Close` | 15 |
| Volatility | `volatility_Nd` (N=5,10,20) | `std(return_1d[t-N+1..t])` | N+1 |
| Volume | `volume_ratio_Nd` (N=5,20) | `Volume / SMA(Volume,N)` | N |
| Volume | `volume_zscore` | `(Volume-SMA(Volume,20))/std(Volume,20)` | 20 |
| Volume | `volume_trend` | `SMA(Volume,5)`の5日間の分数変化率(sma_slopeと同じ式) | 10 |
| RSI | `rsi_7` / `rsi_14` | Wilder方式(初期値=最初のN件のgain/lossの単純平均、以降は再帰平滑化)。avg_gain=avg_loss=0のとき50、avg_loss=0かつavg_gain>0のとき100、avg_gain=0かつavg_loss>0のとき0 | N+1 |
| MACD | `macd` | `EMA(Close,12) - EMA(Close,26)` | 26 |
| MACD | `macd_signal` | `EWM(MACD,span=9,adjust=False)`、最初の33行はNaNに強制マスク | 34 |
| MACD | `macd_hist` | `macd - macd_signal` | 34 |
| Breakout | `high_Nd` (N=5,10,20,60,120) | `max(Close[t-N..t-1])` **(今日のCloseを含まない)** | N+1 |
| Pullback | `distance_from_recent_high` | `Close/max(Close[t-W+1..t]) - 1`(今日を含む、W=`recent_high_window`、config可変・既定20) | W |
| Pullback | `pullback_depth` | `-distance_from_recent_high` (常に0以上) | W |
| Pullback | `distance_from_sma5` / `distance_from_sma20` | `Close/SMA_N - 1` (Trendの`close_to_sma_N`と同一式。Pullback側のSignalがTrend側に依存せず参照できるよう、あえて別カラムとして保持) | 5 / 20 |
| Pullback | `consecutive_down_days` | 直近の連続下落日数(`Close[d]<Close[d-1]`が連続した日数、非該当でリセット)。1行目(前日終値なし)はNaN | 2 |
| Relative Strength | `rs_5d` / `rs_20d` / `rs_60d` | `return_Nd(stock) - return_Nd(market)` (下記「Relative Strength」参照) | N+1(市場側に日付欠損があればそれ以上) |

数値安定性: 0除算・std=0・volume=0等は全て`features/_utils.py::safe_divide`
経由で明示的にNaNへ(infは出しません)。`tests/test_features_pipeline.py::test_no_feature_column_contains_infinite_values`
で全カラムを横断的に検証しています。

## Relative Strength(Phase 3)

### 1306.T = TOPIX Proxy、TOPIX指数そのものではない

`config/settings.yaml`の`data.market_index`に、シンボルだけでなく
`name: "TOPIX Proxy"` / `type: "ETF_PROXY"` を明示的に記録しています。
これは`providers/base.py`の`MarketIndexMeta`(name/symbol/type/source)
として`MarketIndexProvider.describe()`から取得可能で、
`pipeline/build_features.py`のログにも
`Relative Strength benchmark: TOPIX Proxy (1306.T, type=ETF_PROXY, source=yfinance)`
のように出力されます。「TOPIX」という短縮呼称はコード・ログ上どこにも
現れません。

### RSの数学的定義: 超過リターン(比率方式は不採用)

```
rs_5d  = return_5d(stock)  - return_5d(market)
rs_20d = return_20d(stock) - return_20d(market)
rs_60d = return_60d(stock) - return_60d(market)
```

`return_Nd`はPhase 2の`features/momentum.py::compute_return()`と
**完全に同一の関数**を個別株・市場の両方に適用しています(RS専用の
return計算式は存在しません)。`Stock Return/Market Return`のような
比率方式は採用していません(Market Returnが0近傍のときinfや極端な値が
出るため)。市場側の20日リターンが0(横ばい)でもRSは単に
`stock_return_20d - 0`となり、破綻しません
(`tests/test_features_relative_strength.py::test_case4_*`で検証)。

### Date Alignment(行番号ではなく日付でjoin)

個別株と市場ベンチマークで営業日が完全に一致しない可能性(上場直後・
売買停止・ETF側のデータ欠損等)を考慮し、`features/relative_strength.py`
では**行番号による単純な引き算を一切行わず**、以下の手順を踏みます。

1. 市場のCloseを`date`をキーとしたSeriesに変換し、日付でソート
2. 市場自身の時系列だけからmarket_return_Ndを計算(個別株の行順・
   行数には一切依存しない)
3. 個別株の`date`列で`reindex()`し、対応する市場日を引き当てる
   (`method=`引数は渡さない = forward fillしない)
4. 個別株に対応する市場日が存在しない場合、その日のRSはNaN

これにより、市場データの行順を変えても(`test_market_row_order_does_not_affect_aligned_result`)、
市場側に欠損日があっても未来日の値で埋められることなく単純にNaNになる
(`test_missing_market_dates_give_nan_not_forward_filled`)ことをテストで
保証しています。

### Fallback: 市場データが取得できない場合

`providers/market_index.py`の`topix_available`相当の状態
(`data/processed/TOPIX.parquet`が存在しない)を
`pipeline/build_features.py`が検出すると、

- パイプライン全体は停止しない
- warning logを出す
- `BuildFeaturesSummary.market_data_available = False`
- `rs_5d`/`rs_20d`/`rs_60d`は全てNaN(0への置換は一切しない)
- Trend/Momentum/Volatility/Volume/Indicators/Breakout/Pullbackは
  通常通り計算される

という挙動になります。

## Signal Architecture(Phase 4)

SignalはOHLCVそのものではなく**Feature Panel**(Phase 2/3の出力)を
入力とし、`triggered: bool`のみを返します。Score(Phase 5)は一切
参照しません。閾値は全て`config/settings.yaml`の`signals`ブロックに
外出しし、コードにハードコードしていません(値は初期仮説であり、
最適化されたものではありません)。

各Signalの実装は`signals/long/*.py` / `signals/short/*.py`に1ファイル
1Signalで対応し、`signals/registry.py`が12個をまとめて管理します。
`signals/pipeline.py::compute_signal_records()`が
`ticker, date, signal_name, direction, triggered, signal_version`の
long format(triggered=True行のみ)を生成します(全銘柄・全日付は
Feature Panel + configから常に再現可能なため、保存ファイルは
「便利なキャッシュ」であり、再現性の根拠ではありません)。

### LONG 6 Signal

| Signal | 条件(既定値) |
|---|---|
| `long_breakout` | `Close > high_20d` かつ `volume_ratio_20d > 1.5` |
| `long_pullback` | `SMA20 > SMA50` かつ `Close > SMA20` かつ `0.03 <= pullback_depth <= 0.15` |
| `long_ma_rebound` | `SMA20 > SMA50` かつ 前日`Close<=SMA20`→当日`Close>SMA20` |
| `long_momentum_continuation` | `return_5d > 0.03` かつ `return_20d > 0.0` かつ `Close > SMA20` |
| `long_volume_breakout` | `return_1d > 0.03` かつ `volume_ratio_20d > 2.0`(`long_breakout`とは独立、高値更新を要求しない) |
| `long_oversold_rebound` | `RSI14 < 30` かつ `Close > 前日Close` |

### SHORT 6 Signal

| Signal | 条件(既定値) |
|---|---|
| `short_breakdown` | `Close < low_20d` かつ `volume_ratio_20d > 1.5` |
| `short_pullback`(戻り売り) | `SMA20 < SMA50` かつ `Close < SMA20` かつ `0.03 <= bounce_depth <= 0.15` |
| `short_ma_rejection` | `SMA20 < SMA50` かつ 前日`Close>=SMA20`→当日`Close<SMA20` |
| `short_momentum_continuation` | `return_5d < -0.03` かつ `return_20d < 0.0` かつ `Close < SMA20` |
| `short_volume_breakdown` | `return_1d < -0.03` かつ `volume_ratio_20d > 2.0` |
| `short_overbought_reversal` | `RSI14 > 70` かつ `Close < 前日Close` |

各Signalの仮説・期待される挙動・既知のリスクは
`research/signal_notes/*.md`(12ファイル)に記載しています。
「このSignalは儲かる」という表現は意図的に避け、あくまで検証対象の
仮説として記述しています。

### LONG/SHORTの非対称性(意図的な設計判断)

12種のうち10種はLONG/SHORTで完全に対称な条件です。ただし以下2点は
Phase 2/3のFeature Set不足を補うためにPhase 4で追加した対応です。

- `short_breakdown`は`low_Nd`(`features/breakout.py`)を必要とします
  が、Phase 2は`high_Nd`のみ実装していました。Signal層でrolling計算を
  独自実装する(既存の「Feature層を再利用しrollingを再実装しない」
  原則への違反)代わりに、`low_Nd`を`high_Nd`と全く同じ構造
  (`close.shift(1).rolling(N).min()`)でFeature層に追加しました。
- `short_pullback`(戻り売り)は`bounce_depth`(`features/pullback.py`)
  を必要とします。同じ理由で`distance_from_recent_low`/`bounce_depth`
  を`distance_from_recent_high`/`pullback_depth`と対称な構造で追加
  しました。

これらはPhase 2の既存機能を破壊しない追加のみで、Phase 1〜3の全テスト
がそのままPASSすることを確認済みです(詳細は「既知の問題」参照)。

### Signal独立性

各Signalは`compute_signal(panel, config) -> pd.Series`という純粋関数で、
他のSignalの実装や出力を一切参照しません。あるSignalのconfigを変更
しても他のSignalの結果に影響しないことを
`tests/test_signals_pipeline.py::test_changing_one_signal_config_does_not_affect_another`
で検証しています。

## Backtest仕様(Phase 4)

### Entry / Exit

```
Signal date = t
Entry       = Open[t+1]
Exit        = Close[t + 1 + hold_days - 1]   (固定保有日数、既定 hold_days=5)

LONG  Return = Exit/Entry - 1
SHORT Return = (Entry-Exit)/Entry
```

**Entryは絶対にClose[t]を使いません。** `backtest/engine.py`は
Signal実装コード(`signals/long/*.py`等)を一切importせず、
`signals/pipeline.py`が生成したSignalRecordとOHLCV(Feature Panel)
だけを入力とします。これにより同じSignal結果を複数のBacktest設定
(hold_daysや重複抑制ルールを変えたもの)で評価できます。

**SHORT Returnの式について**: 指示書のsection 12は文字通りには
`Entry/Exit - 1`と読めますが、この式はsection 20の手計算例
(Entry=100, Exit=95 → Return=+5%)と数値的に一致しません
(`100/95-1`は+5.26%)。標準的な空売りリターンの定義
`(Entry-Exit)/Entry`はsection 20の例と厳密に一致し、LONGの式と対称的
であるため、こちらを採用しています。`backtest/engine.py`にコメントで
明記しています。

### Entryできないケース(Tradeを作らない)

- t+1データが存在しない
- t+1 OpenがNaN
- Exit日まで必要なデータが存在しない(hold_days分の営業日がない)
- Entry Price <= 0
- Exit Price <= 0

理由は`BacktestRunResult.skipped`(`SkippedSignal`のリスト)に記録
されます。

### 同一銘柄・同一Signalの重複

`config.backtest.suppress_overlapping_signals`(既定 `true`)が有効な
間、同一`(ticker, signal_name, direction)`について前回Tradeがまだ
Exitしていなければ新規Tradeを作りません。`false`にすると全Signal発生
ごとに独立したTradeを作ります。

### Metrics(Phase 4で実装、`backtest/metrics.py`)

`n_trades` / `win_rate` / `average_return` / `median_return` /
`total_return`(等金額・複利なしの単純合計) / `gross_profit` /
`gross_loss` / `profit_factor`(勝ちなしは`None`、負けなしは`inf`) /
`expectancy`(Phase 4はポジションサイジングがないため`average_return`
と等価)。Bootstrap信頼区間・permutation testはPhase 6です。

## Score Architecture(Phase 5)

ScoreはSignal発生行(`triggered=True`)についてのみ計算します。
発生していない行に「仮想Score」は付けません(内部計算自体はPanel
全体に対してベクトル化して行いますが、`scoring/pipeline.py`が
最終的にSignal発生行だけを`data/scores/`へ出力します)。

6カテゴリの配点は`config/settings.yaml`の`scoring.weights`に対応し、
合計は必ず100になります。

| カテゴリ | 配点 | LONGで高得点になる条件 | SHORTで高得点になる条件 |
|---|---|---|---|
| Trend | 0-20(4条件×5) | SMA20>SMA50 / Close>SMA20 / SMA20傾き>0 / SMA50傾き>0 | SMA20<SMA50 / Close<SMA20 / SMA20傾き<0 / SMA50傾き<0 |
| Momentum | 0-20(4条件×5) | return_5d>0 / return_10d>0 / RSI14>50 / macd_hist>0 | 全て符号反転 |
| Volume | 0-15(3条件×5) | volume_ratio_20d>1.2 / (return_1d>0 かつ volume_ratio_5d>1.0) / volume_trend>0 | 3条件目は共通(方向非依存)、2条件目のみ`return_1d<0`に変更 |
| Relative | 0-15(rs_5d/20d/60d各0-5) | rsが高いほど高得点(下記mapping) | `-rs`を同じmappingに通す(単純な符号反転ではない、後述) |
| Setup | 0-20(4条件×5) | return_5d>0.05 / volume_ratio_20d>2.0 / SMA20>SMA50>SMA75 / \|return_1d\|/atr_pct>1.0 | 全て符号・不等号反転 |
| Risk | 0-10(2条件×5) | atr_pct<0.05 かつ volatility_20d<0.03(方向に依存しない) | LONGと共通 |

**重要**: LONG/SHORTは単純な符号反転ではありません。例えばVolume
Scoreの1条件目・3条件目は方向非依存(出来高そのものの確認)であり、
2条件目(価格変化との組み合わせ)だけが方向依存です。Trend/Momentum/
Setupは各条件自体を方向別に定義しており、「LONG Scoreを計算してから
100−Xする」ような実装にはなっていません(この非対称性は
`tests/test_scoring_scorer.py`で直接検証しています)。

### Score Mapping: Relative Score(0-15)

`rs_5d`/`rs_20d`/`rs_60d`それぞれを、`config.scoring.relative.thresholds`
(既定 `[0.05, 0.02, 0.0, -0.02, -0.05]`)に基づき0〜5点にmappingし、
3つの合計(0〜15点)とします。

```
LONG:  rs >= 0.05 → 5, >= 0.02 → 4, >= 0.00 → 3, >= -0.02 → 2, >= -0.05 → 1, それ未満 → 0
SHORT: -rsを同じ閾値ラダーに通す(-rs >= 0.05 → 5 等)
```

RSが取得できない場合(NaN)は0点とし、ScoreそのものをNaNにはしません
(`test_relative_score_nan_gives_zero_not_nan`で検証)。

### Setup Score: Signalのtrigger条件そのものを再利用しない

Setup Scoreは各Signalの発生条件より**厳しい閾値**(例:
`long_momentum_continuation`の`return_5d_min=0.03`に対しSetupは
`return_5d_strong=0.05`)、または**Signalのtrigger条件に現れない
情報**(3本のSMAが完全に並んだ状態か、当日の値動きがATRの何倍かという
「値動きの説得力」)を使っています。「発生条件と完全に同一の情報だけで
Scoreを構成しない」というPhase 5の要件を、閾値を厳しくする・別の
特徴量を使う、の両面で満たしています。

## Forward Target / MFE / MAE(Phase 5、研究専用)

`targets/forward_returns.py`はScore Validation専用のモジュールで、
`features/`・`signals/`・`scoring/`のいずれからもimportされません
(`tests/test_target_leakage.py`でAST解析により自動検証)。

```
Forward Return[t,n] = Close[t+n] / Close[t] - 1     (n = 1, 3, 5, 7, 10)
```

**これはBacktest EngineのEntry=t+1 Openとは意図的に別物です。**
Forward ReturnはSignal発生日t自身のCloseを基準にした単純な価格推移で
あり、実際のトレードルールではありません。

MFE(Maximum Favorable Excursion)/ MAE(Maximum Adverse Excursion)も
同様にSignal日のCloseを基準に、window内のHigh/Lowを使って計算します。

```
LONG:  MFE = max(High[t+1..t+n])/Close[t] - 1   MAE = min(Low[t+1..t+n])/Close[t] - 1
SHORT: MFE = 1 - min(Low[t+1..t+n])/Close[t]     MAE = 1 - max(High[t+1..t+n])/Close[t]
```

符号規約: MFEは通常0以上(最良ケース)、MAEは通常0以下(最悪ケース)で、
LONG/SHORTどちらでも同じ読み方になるよう統一しています。

## Score Validation(Phase 5)

Score ValidationはScore Recordに研究専用のForward Targetを**後から
JOIN**して行います。Score自体はForward Targetの存在を一切前提とせず
(依存方向: `OHLCV → targets/forward_returns.py → Score Validation`
であり、逆方向`targets → features/signals/scoring`は存在しません)、
Forward Targetを削除・改変してもScore計算結果は変化しません
(`tests/test_scoring_no_lookahead.py`のTest C)。

### Bucket

固定幅bucket(`0-19`/`20-39`/`40-59`/`60-79`/`80-100`)とquantile
bucket(`Q1`〜`Q5`、母集団を均等分割)の両方を計算できます
(`scoring/validation.py::assign_fixed_buckets`/`assign_quantile_buckets`)。

### Bucket Metrics

`backtest/metrics.py`の`compute_metrics`をそのまま再利用し(実装を
重複させない)、bucket毎に n / win_rate / average_return(=mean_return)
/ median_return / profit_factor / expectancy を計算します。
Forward 1d/3d/5d/7d/10dそれぞれについて計算します。

### 単調性(Monotonicity)

Score bucketが上がるほどForward Returnが改善するかを、

- `check_monotonicity`: 最低bucketから最高bucketまでaverage_returnが
  非減少かどうかのbool(データ不足で2bucket未満しか比較できない場合は
  `None`)
- `monotonicity_correlation`: bucket順位とaverage_returnのPearson相関
  (-1〜+1の簡易指標、真のSpearman検定ではなく、あくまでPhase 5の
  「簡単な指標」要件を満たすためのもの)

の2つで出力します。**単調性だけでScoreの有効性を判定しません**
(Phase 5の方針、`scoring/validation.py`のdocstringに明記)。

### LONG/SHORT・Signal別分析

`pipeline/run_score_validation.py`は`(direction, signal_name)`で
groupbyしてから各forward windowを分析するため、LONG/SHORTは常に分離
され、12種のSignalそれぞれについて個別のBucket Analysis結果が得られ
ます。

## OOS期間の固定(Phase 6)

Phase 6着手時点(2026-08-19)の実データ範囲は
**2022-01-04〜2024-06-28**(7203/6758/9984/8951の4銘柄)でした。
この範囲と`config/settings.yaml`の`validation.walk_forward`
(`train_months=12, validation_months=3, oos_months=3, step_months=3`)
から、**結果を見る前に**以下5つのWindowが機械的に確定しました
(`backtest/walk_forward.py::generate_windows`は日付とconfigのみの
純粋関数で、乱数・shuffle・train_test_splitは一切使用しません)。

| Window | TRAIN | VALIDATION | OOS |
|---|---|---|---|
| 0 | 2022-01-04〜2023-01-04 | 2023-01-04〜2023-04-04 | 2023-04-04〜2023-07-04 |
| 1 | 2022-04-04〜2023-04-04 | 2023-04-04〜2023-07-04 | 2023-07-04〜2023-10-04 |
| 2 | 2022-07-04〜2023-07-04 | 2023-07-04〜2023-10-04 | 2023-10-04〜2024-01-04 |
| 3 | 2022-10-04〜2023-10-04 | 2023-10-04〜2024-01-04 | 2024-01-04〜2024-04-04 |
| 4 | 2023-01-04〜2024-01-04 | 2024-01-04〜2024-04-04 | 2024-04-04〜2024-06-28(データ終端により短縮) |

この表はPhase 6着手時に一度だけ生成し、以降変更していません。
Window 4のOOSはデータ終端(2024-06-28)により本来の3ヶ月より短くなって
いますが、`min_oos_completeness=0.5`(既定)を満たす(実際は約93%)
ため採用しています。

## Walk Forward構造(Phase 6)

`backtest/walk_forward.py::generate_windows(data_start, data_end, config)`
は日付とconfigだけを引数に取る純粋関数で、Signal/Score/Backtestの結果
を一切参照しません。各WindowはTRAIN終端=VALIDATION開始、VALIDATION
終端=OOS開始という厳密な境界を持ち、Window間はSTEP(既定3ヶ月)ずつ
前進します。

**重要な設計判断**: Feature/Signal/Score/Backtestは各Windowごとに
再計算していません。Phase 2〜5の各パイプラインを**銘柄ごとに1回だけ、
全期間に対して**実行し(Phase4/5の`pipeline/run_backtest.py`・
`pipeline/run_score_validation.py`をそのまま再利用)、生成された
Trade Record / Score RecordをWindowの日付範囲で**事後的にフィルタ**
するだけです。これは手抜きではなく、Phase 2〜5のNo-lookaheadテストが
既に「Feature(t)/Signal(t)/Score(t)はt以前のデータにのみ依存する」
ことを証明しているため、全期間で1回計算してから日付で切り出す方法と、
Windowごとに切り詰めたデータで再計算する方法は**結果が完全に一致する
はず**という前提に基づきます。この前提自体を
`tests/test_walk_forward_no_lookahead.py`のTest Bで直接検証しています。

## Statistical Validation(Phase 6)

### Transaction Cost感応度

`config.validation.transaction_cost.tiers`に定義した4段階
(`zero=0bp`, `low=10bp`, `base=30bp`, `high=80bp`、往復ベーシス
ポイント)を、各Signal・各WindowのOOS Trade Returnから機械的に控除
します(`backtest/costs.py::apply_cost`、単純な固定額控除でspread/
market impactのモデル化はしていません)。全4段階のn/win_rate/PF/
expectancyを出力します。

### Bootstrap(`backtest/bootstrap.py`)

各SignalのOOS集計Trade Return(全Window合算、base costで計算)に
対して、**観測されたTradeそのものから復元抽出**で95%信頼区間を推定
します(mean_return/profit_factor/expectancy)。
`n_resamples=10000, seed=42`で固定し、同じ入力なら常に同じ結果になる
ことを`tests/test_backtest_bootstrap.py`で検証しています。
resample数やseedを「都合の良い結果が出るまで」変更することはして
いません。

### Permutation Test(`backtest/permutation.py`)

帰無仮説「Signal発生とForward Returnの間に特別な関係はない」を検定
します。Backtest EngineのTrade Return(Entry=t+1 Open)ではなく、
Phase 5と同じForward Return(`targets/forward_returns.py`、既定
5日)を使用する点に注意してください(意図的にBacktestとは別物、
「Score Validation」節を参照)。母集団はOOS期間内の**全(ticker,date)
組**のForward Return(Signal発生有無を問わない)とし、Signal発生
サブセットの平均が、同サイズの母集団からのランダム抽出と比べて
外れているかを両側検定します。`n_permutations=10000, seed=43`固定。

### Market Regime(`backtest/market_regime.py`)

TOPIX Proxyの60日リターンのみを使用し(Phase1〜3で既に取得済みの
情報のみ、新たな未来情報は一切追加していません)、`BULL`
(>+5%)/`NEUTRAL`/`BEAR`(<-5%)の3区分でOOS Tradeを分類、regime別に
n/expectancyを出力します。実データの評価期間(2022-2024)では
`BEAR`区分に該当する銘柄横断の該当日がありませんでした(TOPIX Proxy
がこの間60日ベースで-5%を下回らなかったため)。

### Multiple Testing・探索的評価であることの明記

12 Signal × 5 Window × 複数forward horizon × 複数metricsを評価して
いるため、多重検定問題が存在します。Phase 6では「p<0.05だから有効」
と自動判定するロジックを実装していません。`backtest/decision.py`は
raw p-valueをそのまま出力し、Signal数・評価Window数も併記します。
本レポート全体が探索的評価であり、確認的(confirmatory)な統計検定
ではないことをここに明記します。

## Signal Decision(Phase 6)

`backtest/decision.py::classify()`が、各Signalについて
`ACCEPT_CANDIDATE` / `REJECT` / `INSUFFICIENT_EVIDENCE`の3分類を機械的
に出力します。**これは人間のレビュー用の「候補」提示であり、自動採用・
自動デプロイは一切行いません。**

判定基準(Phase 6着手時に固定、OOS結果を見た後に調整していません):

- `INSUFFICIENT_EVIDENCE`: OOS trade数が`min_oos_trades`(既定30)
  未満、評価可能なWindowが0、または集計expectancy/PFが計算不能な場合
- `REJECT`: 以下のうち2つ以上に該当する場合 —
  集計OOS expectancy<=0 / 集計OOS PF<1 / OOS PF>1のWindow比率<50% /
  高コスト条件でexpectancy<=0
- `ACCEPT_CANDIDATE`: 以下を**全て**満たす場合 —
  集計OOS expectancy>0 かつ PF>1 / Bootstrap expectancy 95%CIの下限が
  0より大きい(ゼロを跨がない) / Permutation p値<0.10 / 2Window以上で
  評価され、うち50%以上でOOS PF>1 / 高コスト条件でもexpectancy>0
- それ以外は`INSUFFICIENT_EVIDENCE`

SignalとScoreの判定は独立です(Signal=ACCEPT_CANDIDATEでもScoreの
bucket分析が弱い、逆にSignal=REJECTでもScore単体のbucket分析は別途
参照可能、という組み合わせを許容します)。

## Config Hash / Data Hash(Phase 6)

`common/hashing.py::hash_files()`が、`config/settings.yaml` +
`config/universe_filters.yaml`から`config_hash`を、評価に使用した
`data/features/*.parquet`各ファイルから`data_hash`を、それぞれ
sha256で計算し、`WalkForwardReport`に記録します。これにより「どの
設定・どのデータで得られたOOS結果か」を結果から常に遡れます
(`--save-report`でJSON保存時にも含まれます)。config_hashは
**ファイル自体**のハッシュであり、テスト等でconfigを in-memory で
上書きしても変化しない点に注意してください(意図的な仕様、
`tests/test_pipeline_run_walk_forward.py`で検証)。

## Phase 6.5: Full Universe OOS Validation

Phase 6は4銘柄(7203/6758/9984/8951)でのOOS検証だった。Phase 6.5では
JPX公式データ(`data_j.xls`、`universe/jpx_master.py`)からPrime+
Standard+Growthの全銘柄を対象に、Phase 6と**同一のFeature/Signal/Score/
Backtest/Walk Forward/統計検証ロジックを一切変更せず**、Universeだけを
約3,700銘柄に拡張して再実行した。

- 実行方法: `python scripts/run_universe_ingest.py`(JPX公式データ取得
  +段階的Fetch、`--limit 100`/`--limit 500`でStage 1/2、引数無しで
  Stage 3=Full Universe)→ `python scripts/run_phase6_5_report.py`
  (Feature/Signal/Score再構築+Walk Forward+Data Integrity report)
- 最終結果: Final Universe 2,755銘柄、5 Window(Phase 6と同一日付)、
  12 Signal中 **11 REJECT / 1 INSUFFICIENT_EVIDENCE / 0 ACCEPT_CANDIDATE**
- 全文レポート: [research/phase6_5_full_universe_report.md](research/phase6_5_full_universe_report.md)
  (Per-Signal表・Cost Sensitivity・Regime別・Score Q1-Q5・Concentration・
  Data Quality・Integrity/Hashを含む)
- Full Universe化にあたり、既存コードに**手を入れずには動かせない実バグ
  3件**を発見・修正した(詳細はレポート§9): (1) Universe構築時の
  look-ahead(`universe/filters.py`が流動性判定に最新データを使っていた
  - `.head()`化で修正)、(2) 環境要因のCA証明書パス問題(日本語パスで
  curl_cffiが証明書を読み込めずFull Universe取得が全滅していた -
  コード変更なし、環境変数のみ)、(3) Permutation Testのメモリスケーリング
  (`backtest/permutation.py`が母集団×順列数で約64GBを要求してクラッシュ
  していた - 統計的に同一の結果を保つチャンク処理に変更、rng消費順序が
  同一であることをテストで直接検証)。
- Survivorship Bias: 無料の公式JPXデータでは過去の上場廃止銘柄を再構成
  できないため、本検証は"Current Universe"方式(現在上場している銘柄を
  過去に遡って評価)。J-Quants等の有料データへの登録は行っていない。
- 追加された分析: ticker別breakdown・Concentration(Top1/5/10)・
  Score Q1-Q5 quantile bucket(spread/ratio/bootstrap CI)・
  Benjamini-Hochberg FDR補正(情報提供のみ、Decision判定には不使用)・
  Data Integrity funnel report(`pipeline/data_integrity.py`)。
- Signal条件・Score重み・閾値・HOLD_DAYS・WFO設定・Decision判定基準は
  Phase 6から一切変更していない。0 ACCEPT_CANDIDATEという結果を理由に
  新規Signalを追加する対応も行っていない(仕様のCase G、意図的な方針)。

## Phase 7: 完全独立OOS検証(2024-07-01〜2026-08-20)

Phase 6.5(2022-01〜2024-06)とは完全に無重複の未来期間で、既存12 Signalが
「未知の期間でも再現するか」を検証した。**Signal・Score・Backtest・WFO
設定は一切変更していない**(config_hashがPhase 6.5と完全一致することで
証明済み)。データはPhase 6.5のものを一切上書きせず、`data/phase7/`配下に
完全分離して保存した。

- 実行方法: `python scripts/run_universe_ingest.py --start-date 2022-01-04
  --end-date 2026-08-20 --raw-dir data/phase7/raw --processed-dir
  data/phase7/processed --manifest data/phase7/_universe_fetch_manifest.json`
  → `python scripts/run_phase7_report.py`
- 最終結果: Final Universe 2,880銘柄、9 Window(OOS開始が2024-07-01以降、
  `pipeline/run_walk_forward.py`の新オプション`min_oos_start`でPhase 6.5期
  のWindowを機械的に除外)、12 Signal中 **11 REJECT / 0 INSUFFICIENT_EVIDENCE
  / 1 ACCEPT_CANDIDATE**(`long_oversold_rebound`)
- 全文レポート: [research/phase7_final_report.md](research/phase7_final_report.md)
  (Per-Signal表・Cost Sensitivity・Regime別・Score Q1-Q5・Concentration・
  Phase 6.5 vs Phase 7比較・Case分類・Integrity/Hashを含む)
- **最重要所見**: `long_oversold_rebound`はPhase 6.5のINSUFFICIENT_EVIDENCE
  からPhase 7でACCEPT_CANDIDATEに変わった(PF(base) 1.16→1.80、High cost
  tierでもPF>1、FDR補正後も有意)。ただしRegime別分析で、この結果は
  ほぼ全面的にBEAR regime(PF=42.8、n=1,150)に依存しており、BULL/NEUTRAL
  ではほぼ優位性が消失することが判明した。**この時点でも実運用への自動
  採用は行っていない** - Phase 6.5・Phase 7両方での再現確認後、人間による
  採用判断が必要という仕様上の方針を厳守している。
- Phase 7で発見したバグ(詳細はレポート§9): `pipeline/universe_ingest.py`
  がTOPIX Proxy市場指数を一度も取得していなかった(Phase 6.5は
  `data/processed/`にPhase 1由来の既存TOPIXファイルが偶然存在していたため
  気づかれずに動作していた)。Phase 6.5以前の結果への影響はなし
  (正しいデータで計算されていた)。修正して`run_universe_ingest()`にも
  市場指数フェッチを追加。
- Signal条件・Score重み・閾値・HOLD_DAYS・Backtest条件・WFO設定・Decision
  判定基準は一切変更していない。`long_oversold_rebound`の好結果を理由に
  条件を調整する対応も行っていない。

## Phase 8: long_oversold_rebound 再現性・BEAR regime依存性の追加検証

唯一ACCEPT_CANDIDATEとなった`long_oversold_rebound`について、その優位性が
「特定期間・特定銘柄への偶然の依存ではないか」「BEAR regimeでのみ発生する
一時的な現象ではないか」を追加検証した。**Signal・Score・Backtest・WFO・
Cost・Bootstrap・Permutation・Decision Frameworkは一切変更していない**
(config_hashがPhase 6.5・Phase 7・現在の3値で完全一致することを機械的に
確認済み)。データはPhase 7が既に取得済みの`data/phase7/`(2022-01-04〜
2026-08-20の連続データセット)をそのまま再利用し、新規Fetchは行っていない。

- 実行方法: `python scripts/run_phase8_analysis.py`
- 全文レポート: [research/phase8_report.md](research/phase8_report.md)
  (Combined OOS・BEAR×Window・BEAR×年別・BEAR episode別・Leave-One-Period-Out・
  Placebo negative control・銘柄集中度・Cost/Bootstrap/Permutation・他11
  Signal簡易比較を含む)
- **最重要所見**: Combined OOS(Phase 6.5+Phase 7通し、全14 Window)では
  PF(base)=1.585でACCEPT_CANDIDATE。BEAR regime限定PFは34.9と極めて高いが、
  **その累積リターンの71.6%が2024年8月の1回の市場急落・急反発イベント
  (9営業日のみ)に集中している**ことが判明した。このepisodeを除いても
  PFは1を割らない(39.6→14.9)が、62%の低下は無視できない。銘柄集中は
  確認されず(Top20銘柄でtrade share 5.3%)、Placebo negative control
  (同一Signalを15営業日前にずらして再実行)ではBEAR regime PFが0.095と
  実際のSignalの1/400以下になり、タイミングそのものに意味があることを
  支持した。
- 機械的な基準だけならCase R1(ROBUST_REGIME_DEPENDENT)の条件を満たすが、
  単一イベントへの集中という重大な留保付きで結論している
  (`ROBUST_REGIME_DEPENDENT with major single-episode concentration
  caveat`)。**実運用への自動採用は行っていない。**
- Phase 8で個別銘柄の生OHLCVを直接確認し、極端なリターンが株式分割等の
  データ異常ではなく実際の2024年8月の市場イベントであることを検証済み
  (バグではない)。

## Phase 9: long_oversold_rebound 独立ロバストネス検証

Phase 8で判明した「BEAR regimeの累積リターンの71.6%が9営業日に集中」
という所見を受け、「2024年8月急落・急反発という特殊イベントへの依存か、
一般化可能なregime-dependent edgeか」を、有効性を崩す条件を積極的に
探す形で検証した。**Signal・Score・Backtest・WFO・Cost・Decision
Frameworkは一切変更していない**(config_hashがPhase 6.5・7・現在の3値で
完全一致)。データはPhase 7/8が既に取得済みの`data/phase7/`をそのまま
再利用し、新規Fetchは行っていない。

- 実行方法: `python scripts/run_phase9_analysis.py`
- 全文レポート: [research/phase9_report.md](research/phase9_report.md)
  (Episode拡張分析・Leave-One-Episode/Year-Out・Day Cluster/Block
  Bootstrap・Timing Placebo Sweep・Sector/Liquidity別分析・Cost Stress・
  事前固定シナリオA-F・Forward Holding Period感応度を含む)
- **最重要所見**: BEAR regime限定の優位性は、たった**1営業日で全損益の
  64.8%**(Gini係数0.956)という、Phase 8の想定以上に極端な集中を示した。
  Timing Placebo Sweep(実際のSignalタイミングを-15〜+10営業日ずらして
  再実行)では、数営業日離れるだけで優位性が急速に崩壊することを確認。
  一方、その支配的な2024年8月episodeを完全に除外してもBEAR限定でHigh
  cost tierでもPF=10超を維持し、より保守的な統計手法(Day Cluster/Block
  Bootstrap)でもBEAR PFの信頼区間下限は1を明確に上回ったままだった。
  Combined(全regime込み)は6つの事前固定ストレスシナリオ全てでPF>1を
  維持した。
- **最終分類**: `Primary: EVENT_DEPENDENT` / `Secondary caveat:
  REGIME_DEPENDENT`。特定イベントへの依存という側面と、そのイベントを
  除いても残るregime条件付きのエッジという側面の両方が確認された。
  **実運用への自動採用は行っていない。**
- Phase 9で新たに発見したコード上のバグはなし。

## Phase 10: Frozen Strategy Forward Test Engine

`long_oversold_rebound`を完全凍結し、過去データの分析ではなく未来の
市場に対するPaper Trading(Forward Test)基盤を構築した。**Signal・
Score・Backtest・WFO・Cost・Regime定義は一切変更していない。**

- 実行方法: `python scripts/run_forward_test_day.py`(初回実行時に
  T0を確定してStrategy manifestを自動生成、以降は営業日ごとに再実行
  することで日次更新)
- 全文レポート: [research/phase10_report.md](research/phase10_report.md)
- **T0 = 2026-08-20**。Strategy code(features/signals/scoring/
  backtest engine/market regime)と既存config_hashの両方をSHA256で
  凍結し、以後1つでも変化があれば`StrategyHashMismatchError`で即座に
  停止する仕組みを実装(仕様に基づき、コード変更が必要な場合は
  Strategy Version 2として完全に別管理する設計)。
- Paper Portfolio(実資金は一切使用しない仮想ポートフォリオ)・
  Signal Log(追記専用、後からの再計算・上書きは不可)・Data/Trading
  Integrity監視を実装し、T0の実データでend-to-end動作確認済み
  (Final Universe 2,780銘柄、当日Signal 4件検出、Portfolio equity
  10,000,000円で初期化)。
- **実際のForward Test結果が蓄積するまでは、戦略の有効性について一切
  結論を出していない。** 最低6か月間はstrategy tuningを行わない方針。
- 証券会社API接続・自動発注・実資金投入は一切実装していない
  (Paper Tradingのみ)。

## Phase 11: Forward Test自動日次実行 + 残り11 Signal独立検証

明示的に独立した2つのTrackで構成される。**一方の結果は他方に一切
影響を与えない。** 全文レポート:
[research/phase11_report.md](research/phase11_report.md)

**Track A(Forward Test自動日次実行)**

- Phase 10のFrozen Strategy Forward Test Engineを拡張し、SAFE_ABORT
  (MARKET_DATA_UNAVAILABLE / UNIVERSE_DATA_INCOMPLETE /
  STALE_THRESHOLD_EXCEEDED / FEATURE_GENERATION_FAILURE /
  SIGNAL_GENERATION_FAILURE / PORTFOLIO_STATE_CORRUPTION)・Open
  Position追跡(未決済ポジションの含み損益)・Daily Performance Log
  (追記専用)を実装。Signal・Score・Backtest・凍結対象は一切変更して
  いない。
- GitHub Actions定期実行は本ディレクトリがgitリポジトリ化されて
  いないため未実装。仕様が許容する代替として、冪等設計を活かした
  安全なCLI再実行(`python scripts/run_forward_test_day.py`)を採用。
- 実データで2026-08-20(T0)の冪等性・2026-08-21のSAFE_ABORT
  (STALE_THRESHOLD_EXCEEDED、市場データ2,782/2,782銘柄がstale)を
  確認済み。Forward Test結果はまだ2営業日分のみで、**戦略の有効性
  について一切結論を出していない。**

**Track B(残り11 Signal独立検証)**

- `long_oversold_rebound`以外の11 Signal(long_breakout / 
  long_ma_rebound / long_momentum_continuation / long_pullback /
  long_volume_breakout / short_breakdown / short_ma_rejection /
  short_momentum_continuation / short_overbought_reversal /
  short_pullback / short_volume_breakdown)を、Phase 6.5-9と同等以上
  の厳密さ(Bootstrap・Permutation Test・FDR多重検定補正・BEAR
  Placebo・Timing Placebo・Event concentration等)で独立検証した。
- Full Universe(2,880銘柄、data/phase7/)で実施。**結果: 11 Signal
  全てREJECT。** ACCEPT_CANDIDATEはゼロ。最小raw p-valueは0.0541
  (FDR補正後q=0.3246)で有意水準に届かず、特定期間・イベントへの
  依存で結論が変わっている様子もない。
- ACCEPT_CANDIDATEが出ていないため、Strategy Version 1への自動反映
  は当然発生していない(そもそも本仕様はACCEPT_CANDIDATEでも自動
  反映を禁止している)。
- 本Phaseの過程で`backtest/bootstrap.py`のメモリスケーラビリティ
  バグ(高頻度発火Signalで35GBの配列確保が発生)を発見・修正した。
  Signal/Score/Decisionロジックには一切手を加えていない、純粋な
  実装バグの修正。

## Phase 12: Signal Ensemble Validation + Extended Forward Targets

既存12 Signalは一切変更せず、(1) Phase 5 Forward Target構造に15d/20d
を追加、(2) 既存12 Signalの「同時発生(Ensemble)」の統計的妥当性を
検証した。全文レポート:
[research/phase12_report.md](research/phase12_report.md)

- Forward Return/MFE/MAEの評価期間を1/3/5/7/10dから1/3/5/7/10/15/20d
  に拡張(`targets/forward_returns.py::FORWARD_WINDOWS`)。既存
  1/3/5/7/10dの計算結果がbit-identicalであることを直接証明する回帰
  テストを追加済み。
- 新規`ensemble/`パッケージでSignal Count(LONG_COUNT/SHORT_COUNT/
  NET_SIGNAL_COUNT)・Direction Consensus・自然発生した組み合わせ
  (2-way〜4+-way)・出現頻度・Decision Framework
  (ROBUST_ENSEMBLE/REGIME_DEPENDENT_ENSEMBLE/EVENT_DEPENDENT_ENSEMBLE/
  FREQUENCY_TOO_LOW/REJECT/INSUFFICIENT_EVIDENCE)を実装。
- Full Universe(2,880銘柄)で実施。**LONG/SHORT/NET全17 primary
  bucketがREJECT** — 「Signal数が多いほど期待値が上がる」という
  仮説は支持されなかった。自然発生した組み合わせ38件(sufficient
  sample)のうち36件REJECT、残り2件は自動判定ではROBUST_ENSEMBLEと
  出たが、手動フォローアップでいずれもBEAR regime依存性が判明し
  REGIME_DEPENDENT_ENSEMBLEと再評価した(詳細はレポート項目K)。
  **Phase 12終了時点でROBUST_ENSEMBLEと呼べる組み合わせは1つも
  ない。**
- Top-5簡易シミュレーション(固定選択ルール)はtotal_return=-9.1%、
  CAGR=-2.1%とマイナスの結果。
- Strategy Version 1(`long_oversold_rebound`のForward Test)には
  一切影響しておらず、Integrity Hashも開始時・終了時で完全一致を
  確認済み。実運用への自動採用は行っていない。

## Phase 11A: GitHub Actions完全自動Forward Test化

Phase 10/11のForward TestをClaude Codeの実行環境に依存せず、
GitHub Actionsのみで毎営業日自動継続できる状態にした。全文
レポート: [research/phase11a_report.md](research/phase11a_report.md)

- `.github/workflows/forward_test.yml`を新規作成
  (schedule: 平日21:00 JST + workflow_dispatch)。既存の
  `scripts/run_forward_test_day.py`をそのまま起動し、結果の
  append-only状態(Signal Log・Paper Portfolio・Daily Performance
  Log等)をリポジトリにcommit・pushする。SAFE_ABORTは失敗扱いに
  せず明示表示、Strategy Hash不一致等の想定外エラーのみjob失敗
  とする設計。
- 実環境で3回workflow_dispatchを実行し、2件の実装バグを発見・
  修正した:
  1. `common/hashing.py::hash_files()`が`str(path)`(OS依存の
     区切り文字)をハッシュ入力に使用しており、Windows上で計算
     したStrategy HashがLinux(GitHub Actions)では永遠に一致
     しないバグ。`Path.as_posix()`ベースに修正し、Strategy Hash
     対象38ファイルがバイト単位で無変更であることを独立検証した
     上でmanifestを再生成(Strategy Version 1のまま、Version 2
     への移行ではない)。
  2. workflow内の`git add`が、一度もファイルを書き込まれたこと
     のない空ディレクトリ(`data/forward_test/reports/`)に対して
     失敗するパスバグ。`mkdir -p`で修正。
- 3回目の実行でUniverse取得(2,781銘柄)からScore算出まで
  パイプライン全体が正常完了した上で、市場データの当日分未到達を
  正しく検知し`SAFE_ABORT[STALE_THRESHOLD_EXCEEDED]`で安全停止
  したことを確認(実データでの安全機構の実地動作確認)。
- Signal/Score/Backtest/Decision Frameworkのロジックは一切変更
  していない。

## Phase 13: long_oversold_rebound Conditional Analysis

`long_oversold_rebound`(Strategy Version 1)の過去発生行を対象に、
どの市場・銘柄・Score条件でForward Returnが強いかを事後分析した。
既存12 Signal・Score・Backtestは一切変更していない。Signal改良では
なく、独立検証すべき仮説の抽出のみが目的。全文レポート:
[research/phase13_report.md](research/phase13_report.md)

- Full Universe(2,880銘柄)で実施、対象Signal 20,670件
  (unique ticker 2,606、発生日数1,081日)。Regime・Market
  Drawdown(TOPIX 20d return)・個別銘柄Drawdown・MA乖離・出来高・
  Volatility・Score・Signal Count・LONG/SHORT一致度の9軸単体分析と
  Regime×Score・Market Drawdown×Scoreの2 cross-tab分析、Forward
  Horizon(1〜20d)・Event Exclusion・BEAR Episode分析・64ユニット
  へのFDR多重検定補正を実施。
- **主な発見(いずれも探索的、未検証)**: BEAR regimeおよびTOPIX
  20d return -10%以下の局面でPFが大きく上昇する一方、下落してい
  ない局面(全Signalの約半数)ではエッジが消失する。この傾向は
  Score水準にほぼ依存しない(Regime×Score・Drawdown×Score両方で、
  全Scoreクインタイルが同じパターンを示す)。出来高が多いほど期待
  値も単調に強い。Score単体はForward Returnとほぼ無相関
  (monotonicity不成立)。
- 2024年8月イベントを除いてもBEAR局面全体のPFは1を上回り、さらに
  2025年4月の独立した別のBEAR急落局面でも同様の強さ(PF=78.6)が
  再現した — 単一イベント依存ではないが、2つの主要episodeへの
  集中(累積寄与度96.8%)という別の集中パターンが確認された。
- Strategy Version 1のForward Testには一切影響していない。発見した
  仮説を現在のSignal・Score・Forward Test Engineに反映することは
  行っていない。

## No-lookahead対策

構造面: 全Feature関数は`rolling()`・`ewm()`・`shift(+n)`のみを使用し、
`shift(-n)`や未来行を参照する処理は一切使用していません
(Breakoutの`high_Nd`は`close.shift(1).rolling(N)`という並び順が
必須 - 先にrollingしてから比較すると今日自身の値が閾値に混入します)。

テスト面(`tests/test_no_lookahead.py`、Phase 2の指示に沿った4種。
Phase 3でTest A/BはRelative Strengthも対象に含むよう拡張し、
市場データ側の未来改変を検証するRS専用のTest Cを追加):

- **Test A (Truncation Test)**: 同じ銘柄(+市場ベンチマーク)をt日までの
  データセットとt+100日以降まで含むデータセットの両方で計算し、
  Feature(t)(rs_*含む)が完全一致することを検証(t=100,150,200,250,290)。
- **Test B (Future Perturbation Test)**: t+1日以降の個別株OHLCVを
  ランダムに±50%変更してもFeature(t)(rs_*含む)が変化しないことを検証。
- **RS Test C (Market Future Perturbation Test)**: 個別株は固定し、
  t+1日以降の**市場ベンチマーク**をランダムに変更してもrs_Nd(t)が
  変化しないことを検証(他Featureにはこの依存経路自体が存在しないため
  RS専用)。
- **Test C (Feature Dependency Test)**: `features/*.py`が
  `targets`/`forward_returns`および`providers`を一切importしていない
  ことをASTベースで自動検証。
- **Test D (Mathematical Property Test)**: 単調増加/減少系列での
  SMA傾きの符号、一定価格系列でのreturn/volatility/RSI/MACDの値、
  Stock+10%/Market+5%→RS+5%などRS超過リターンの数学的性質を検証
  (`tests/test_features_relative_strength.py`)。市場データの行順に
  依存しないこと(RS版Test D)もここで検証。

Backtest専用(`tests/test_backtest_no_lookahead.py`、Phase 4指示の
section 19に対応):

- **Test A**: Signal(t)を計算後、t+1以降のOHLCVをランダムに変更しても
  Signal(t)(全12種)が変化しないこと。
- **Test B**: Signal Recordを固定したまま、t+1 Openだけを変えると
  Trade Returnは変わるが、Trade化される日付・件数(=Signal判定)自体は
  変わらないこと。
- **Test C**: 実際に生成された全TradeについてEntry Priceが
  厳密に`Open[t+1]`と一致することをランダムなシグナル群で検証。
- **Test D**: Signal発生日のCloseに`999999`という明白な異常値を仕込み、
  Entry Priceがその値に**絶対に**汚染されないことを検証。

Score専用(`tests/test_scoring_no_lookahead.py`、Phase 5指示の
section 24に対応):

- **Test A**: t+1以降の個別株OHLCVをランダムに変更してもScore(t)
  (6サブスコア+total_score)が変化しないこと。
- **Test B**: t+1以降の市場ベンチマークをランダムに変更しても、
  Relative Scoreを含むScore(t)が変化しないこと。
- **Test C**: Forward Return/MFE/MAEを計算した後に値を破壊
  (`999999`等)しても、既に計算済みのScoreが変化しないこと
  (Scoreの計算経路にForward Targetへの参照が構造的に存在しないことの
  動的な確認。静的な確認は下記Target Leakage Test)。

Target Leakage Test(`tests/test_target_leakage.py`、Phase 5指示の
section 25に対応): `features/`・`signals/`・`scoring/`配下の全`.py`
ファイルをASTで解析し、`targets`(またはそのサブモジュール)への
importが1件も存在しないことを検証。逆方向(`targets/`が
`features`/`signals`/`scoring`に依存していないこと)も同時に検証して
います(`pipeline/run_walk_forward.py`・`backtest/permutation.py`は
Score Validationと同じ「検証層」として`targets`を利用しますが、
`features`/`signals`/`scoring`には含まれないため、この制約の対象外
です)。

Walk Forward専用(`tests/test_walk_forward_no_lookahead.py`、Phase 6
指示のsection 31に対応):

- **Test A**: あるOOS Window開始日より未来のOHLCVをランダムに変更
  しても、そのOOS開始日より前のSignal Recordが変化しないこと。
- **Test B**: 全期間で1回計算してからTRAIN期間で日付フィルタした
  結果と、TRAIN終端で切り詰めたデータセットから再計算した結果が
  完全一致すること(「Walk Forward構造」節で説明した設計上の前提の
  直接検証)。
- **Test C**: `validation_months`だけを変更したとき、TRAIN境界は
  一切変わらず、OOS開始日が変更分だけ厳密に(意図通り)後ろへ
  ずれること(=隠れた結合バグが無いことの確認)。
- **Test D**: Forward Returnの値を破壊してもSignal Recordが変化しない
  こと(Phase 5のTest CをWalk Forwardの文脈で再確認)。

Deterministic Test(`tests/test_backtest_bootstrap.py`・
`tests/test_backtest_permutation.py`・
`tests/test_pipeline_run_walk_forward.py`、Phase 6指示のsection 32に
対応): 同一data・同一config・同一seedならBootstrap CI・Permutation
p値・`run_walk_forward()`の全出力(decision・cost tier別metrics・
bootstrap・permutationを含む)が完全に再現することを検証しています。

## ディレクトリ構造(Phase 6時点)

```
swing-scanner/
    config/
        settings.yaml            # 全設定の単一ソース
        universe_filters.yaml    # 価格・流動性フィルタ(動的、fetch後に適用)
        loader.py                # YAML→pydanticモデル
    common/
        logging_setup.py         # ロギング初期化
        hashing.py                 # config_hash/data_hash計算(sha256、Phase6)
    providers/
        base.py                  # OHLCVProvider/MarketIndexProvider抽象IF、MarketIndexMeta
        yfinance_provider.py     # yfinance実装(retry/backoff付き)
        market_index.py          # TOPIX Proxy等の指数取得(yfinance実装、describe()でmeta公開)
    universe/
        build.py                 # 静的フィルタ(市場区分・ETF/REIT除外)
        filters.py                # 動的フィルタ(価格・流動性、fetch後に適用)
    validation/
        ohlcv.py                 # データ品質検証(黙って修正しない)
    features/
        _utils.py                 # sma/ema/slope/safe_divide等の共通関数(非公開)
        metadata.py                # FeatureMeta(formula/warmup等の宣言的メタデータ)
        trend.py                   # Trend + MA Distance
        momentum.py                 # Return(compute_returnをRSと共有)
        volatility.py               # ATR(Wilder)・volatility
        volume.py                   # volume_ratio/zscore/trend
        indicators.py               # RSI(Wilder)・MACD
        breakout.py                  # high_Nd/low_Nd(今日を除く過去N日高値・安値)
        pullback.py                  # 押し目・戻り系Feature、recent_high_windowが可変
        relative_strength.py         # rs_5d/20d/60d(date alignment、市場None時はNaN)
        pipeline.py                  # 全カテゴリを結合してFeature Panelを生成
    signals/
        base.py                    # Direction、SignalMeta、require_columns
        registry.py                  # 12 Signalの一覧(SIGNAL_REGISTRY)
        pipeline.py                   # compute_signal_panel/compute_signal_records
        long/
            breakout.py, pullback.py, ma_rebound.py,
            momentum_continuation.py, volume_breakout.py, oversold_rebound.py
        short/
            breakdown.py, pullback.py, ma_rejection.py,
            momentum_continuation.py, volume_breakdown.py, overbought_reversal.py
    backtest/
        engine.py                  # run_backtest_for_ticker(Signal→Trade、Entry=t+1 Open)
        metrics.py                   # win_rate/profit_factor/expectancy等(scoringからも再利用)
        walk_forward.py               # TRAIN/VAL/OOS Window生成(純粋関数、乱数不使用)
        costs.py                       # Transaction Cost 4段階の適用
        bootstrap.py                   # Bootstrap信頼区間(ベクトル化、seed固定)
        permutation.py                 # Permutation Test(Forward Return対象)
        market_regime.py               # BULL/NEUTRAL/BEAR判定(TOPIX Proxy 60日リターンのみ)
        decision.py                    # ACCEPT_CANDIDATE/REJECT/INSUFFICIENT_EVIDENCE候補判定
    scoring/
        scorer.py                  # 6カテゴリScore(Trend/Momentum/Volume/Relative/Setup/Risk)
        pipeline.py                  # compute_score_records(Signal発生行のみにScore付与)
        validation.py                 # bucket割当・bucket metrics・単調性判定(Phase6でも再利用)
    targets/
        forward_returns.py         # Forward Return/MFE/MAE(研究専用、features/signals/scoringからimport禁止)
    storage/
        parquet_store.py         # Parquet読み書き(feature panel/signal/score records用I/Oも含む)
    pipeline/
        ingest.py                # Phase 1オーケストレーター
        build_features.py        # Phase 2/3オーケストレーター(processed→features、TOPIX Proxy読込)
        build_signals.py         # Phase 4オーケストレーター(features→signals)
        run_backtest.py          # Phase 4オーケストレーター(signals+features→trades)
        build_scores.py          # Phase 5オーケストレーター(features+signals→scores)
        run_score_validation.py  # Phase 5オーケストレーター(scores+targets→bucket analysis)
        run_walk_forward.py      # Phase 6オーケストレーター(Phase4/5を再利用しWindow別に集計)
    data/
        raw/                      # provider取得直後(監査用、未加工)
        processed/                # 検証・クリーニング後
        processed/universe/       # 日次のuniverseスナップショット
        features/                 # Phase 2/3で生成するFeature Panel
        signals/                  # Phase 4で生成するSignal Record(triggered=Trueのみ)
        scores/                   # Phase 5で生成するScore Record(triggered=Trueのみ)
        backtest/                 # Backtestで生成するTrade Record(任意保存)
        walk_forward/              # Phase 6のJSONレポート(任意保存、config_hash/data_hash含む)
        reference/                # Universe元データ(下記「既知の問題」参照)
    research/
        signal_notes/             # 12 Signalの仮説ドキュメント+Phase 6 OOS評価結果(Markdown)
    scripts/
        run_ingest.py             # Phase 1 CLIエントリポイント
        run_build_features.py     # Phase 2/3 CLIエントリポイント
        run_build_signals.py      # Phase 4 CLIエントリポイント(Signal)
        run_backtest.py           # Phase 4 CLIエントリポイント(Backtest)
        run_build_scores.py       # Phase 5 CLIエントリポイント(Score)
        run_score_validation.py   # Phase 5 CLIエントリポイント(Score Validation)
        run_walk_forward.py       # Phase 6 CLIエントリポイント(Walk Forward)
    tests/
```

Streamlit UI等は今後のPhaseで追加します(既存の設計ドキュメントの
Phase 7〜8を参照)。

## 既知の問題 / 未確認事項

1. **Universe master listはサンプルデータです。**
   `data/reference/jpx_listed_companies.sample.csv` は開発・テスト用に
   手作業で作成した15銘柄のみのプレースホルダーであり、実際の東証全上場
   銘柄リストではありません。本番運用にはJPX公式サイト([東証上場銘柄一覧](https://www.jpx.co.jp/markets/statistics-equities/misc/01.html))
   から取得したファイルを `code,name,market_segment,sector33` の列構成に
   整形し、`config/settings.yaml` の `universe.master_list_path` が指す
   場所に配置する必要があります。

2. **TOPIXは実際のindexシンボルでは取得できません(検証済み・2026-08-19)。**
   `^TOPX` / `^TPX` / `TOPX` / `998405.T` はいずれも空データが返ります。
   このyfinance/Yahoo Financeフィードでは生のTOPIX指数値そのものを取得
   できないため、代替として `1306.T`(野村アセットのTOPIX連動型ETF、
   「TOPIX Proxy」)をRelative Strengthのベンチマークとして使用して
   います(`config/settings.yaml` の `data.market_index`)。ETFである
   ため、指数値そのものとは厳密には一致しません(信託報酬・トラッキング
   誤差・分配金落ちの影響を受けます)が、リターン系列としてはTOPIXに
   極めて近く追随します。より正確な指数フィードが必要になった場合
   (例: J-Quants導入時)は、このconfig値(`ticker`/`name`/`type`)と
   `MarketIndexProvider`実装を差し替えるだけで、
   `features/relative_strength.py`のRS計算ロジックには一切手を
   入れずに済む設計になっています(Phase 3で実装・確認済み)。TOPIX
   Proxy取得が失敗した場合はパイプラインが自動的にRelative Strengthを
   無効化し、ログに警告を出します(`topix_available`
   フラグ→`BuildFeaturesSummary.market_data_available`として
   Phase 3で配線済み)。

3. **作業ディレクトリのパスに日本語などの非ASCII文字が含まれる場合、
   yfinanceの内部HTTPクライアント(curl_cffi)がCA証明書ファイルを
   読み込めず、全てのfetchが `possibly delisted` という紛らわしい
   エラーで失敗することを確認しました。** この開発環境
   (`...\デスクトップ\claude-work`)でも実際に発生し、環境変数
   `CURL_CA_BUNDLE` と `SSL_CERT_FILE` を非ASCII文字を含まないパスに
   コピーした `cacert.pem` を指すよう設定することで解消しました。
   ```bash
   cp .venv/lib/site-packages/certifi/cacert.pem /tmp/cacert.pem
   export CURL_CA_BUNDLE=/tmp/cacert.pem
   export SSL_CERT_FILE=/tmp/cacert.pem
   ```
   恒久対応としては、プロジェクトをASCII文字のみのパスに置くか、上記の
   環境変数をシェルプロファイルに設定することを推奨します。アプリ
   ケーションコード側では対応していません(環境固有の問題であり、
   コードに埋め込むべきではないため)。

4. **取引カレンダーは実際のJPX営業日カレンダーではなく、pandasの
   business day(`bdate_range`)で近似しています。** 祝日を営業日として
   カウントするため、SUCCESS/PARTIAL判定のカバレッジ計算がわずかに
   厳しめに出ます。実データでの検証時に閾値(`min_expected_coverage`)の
   調整が必要になる可能性があります。

5. Universe構築は「静的フィルタ(市場区分・ETF/REIT、master listのみで判定)」
   と「動的フィルタ(価格・流動性、fetchしたOHLCVが必要)」の2段階です。
   静的フィルタはfetch前、動的フィルタはfetch後に適用されます
   (`universe/build.py` と `universe/filters.py` の分離はこの理由による)。

6. **MA DistanceはTrendと同じファイル(`features/trend.py`)に実装しています。**
   設計ドキュメントのカテゴリ分けでは別区分ですが、`close_to_sma_N`は
   Trendが計算するSMA値をそのまま使う以外に定義しようがないため、
   ファイル分割よりも依存関係の凝集度を優先しました。

7. **`distance_from_sma5`/`distance_from_sma20`(Pullback)は
   `close_to_sma_5`/`close_to_sma_20`(Trend)と数式上は同一です。**
   意図的な重複です。Pullback系のSignal(Phase 4)がTrend特徴量群への
   依存なしに参照できるよう、Pullbackカテゴリ側にも同じ値を独立した
   カラムとして持たせています。

8. **RSIはpandasの`ewm(alpha=1/period)`による近似ではなく、Wilderの
   元の定義(初期値=最初のperiod件の単純平均、以降は再帰平滑化)を
   明示的に実装しています。** MACDのシグナル線・生のEMA(ema_5/20/50)は
   pandasの`ewm(adjust=False)`をそのまま使いつつ、この項目のwarmup規約
   (最初のspan-1行はNaN)に合わせて明示的にマスクしています。
   pandasのewmはデフォルトで1行目から値を返す(初期の値は情報量が
   少なく信頼性が低い)ため、そのまま使うとSMA系Featureとwarmupの
   意味がずれるからです。

9. **RSのwarmup_period(N+1)は「市場ベンチマークが個別株の日付範囲を
   完全にカバーしている」という前提の理論上の最小値です。** 実データで
   市場側に欠損期間(ETFの売買停止等)があると、その期間だけ実際の
   NaN数が宣言値を上回ります。これはバグではなく仕様です
   (`features/relative_strength.py`のdocstring参照)。そのため、
   `test_feature_warmup_matches_declared_metadata`はrs_*についても
   厳密なNaN数一致を検証していますが、これは市場データが個別株と
   完全に日付一致する合成データでのみ成立する検証である点に注意して
   ください。

10. **SHORT Returnの計算式は指示書の記述と実装が異なります(意図的)。**
    section 12の文字通りの式`Entry/Exit-1`ではsection 20の手計算例
    (Entry=100,Exit=95→+5%)を再現できないため、その例と厳密に一致し
    LONGと対称的な`(Entry-Exit)/Entry`を採用しました。詳細は
    「Backtest仕様」節を参照してください。

11. **`low_Nd`(`features/breakout.py`)と`distance_from_recent_low`/
    `bounce_depth`(`features/pullback.py`)はPhase 4開始時に追加した
    Feature層への拡張です。** Phase 2はLONG向けの`high_Nd`/
    `pullback_depth`のみを実装しており、SHORT Signal
    (`short_breakdown`/`short_pullback`)を「独自にrolling計算を
    再実装しない」という原則を保ったまま実装するには、対称な
    Feature側の追加が必要でした。既存の`high_Nd`/`pullback_depth`の
    計算・warmup・no-lookahead特性は一切変更していません
    (Phase 1〜3の全既存テストがそのままPASSすることを確認済み)。

12. **実データでのBacktest実行結果(2022-01-01〜2024-06-30、7203/6758/
    9984/8951の4銘柄)は、Signalの有効性を示すものではありません。**
    サンプル数が少なく(最大4銘柄)、対象期間は日本株が総じて上昇した
    局面を多く含むため、LONG系Signalの平均リターンが正、SHORT系が負に
    偏る結果が出ています。これは「パイプラインが技術的に正しく動作する
    ことの確認」であり、Phase 6のOOS検証を経ていない現時点でこの数字を
    根拠にSignalを採用・棄却してはいけません(section 26参照)。

13. **Score bucket分析の`compute_bucket_metrics`は
    `backtest/metrics.py::compute_metrics`をそのまま再利用しています。**
    そのため出力フィールド名は`BacktestMetrics`のもの(例:
    `average_return`)であり、Phase 5指示書の`mean_return`という表記と
    完全一致しません。値の意味は同一(単純平均リターン)であるため、
    別データクラスとして重複定義するより再利用を優先しました。

14. **実データでのScore bucket analysis結果
    (2022-01-01〜2024-06-30、4銘柄、forward=5dで確認)は、単調性が
    一部のSignalでのみ観測され、他では観測されませんでした
    (`monotonic=True`は12種中2種のみ)。** これはサンプル数の少なさ・
    重み/閾値が未最適化の
    初期仮説であることを踏まえれば想定内であり、「Scoreは無効」とも
    「Scoreは有効」とも結論づけられません。Phase 6のOOS検証と統計的
    検定を経るまでは、この結果に基づいて重みや閾値を調整しません
    (Phase 5の禁止事項)。

15. **Phase 6のPermutation Testは`backtest/engine.py`のTrade Return
    (Entry=t+1 Open)ではなく、Phase 5と同じForward Return
    (Signal日tのCloseを基準)を対象にしています。** 「Signal発生と
    Forward Returnの間に特別な関係はない」という指示書section 13の
    文言に忠実に従った結果であり、Bootstrap(Trade Returnが対象)と
    Permutation Test(Forward Returnが対象)で対象とするリターンの
    定義が異なる点に注意してください。両者を同じ「リターン」として
    直接比較できません。

16. **実データでのWalk Forward実行結果(5 Window、12 Signal)は、
    `ACCEPT_CANDIDATE`が0件、`REJECT`が2件
    (`long_momentum_continuation`、`short_momentum_continuation`)、
    残り10件が`INSUFFICIENT_EVIDENCE`でした。** これはサンプル数が
    少ない(4銘柄・OOS期間合計約15ヶ月)ことを踏まえれば想定内の、
    健全に保守的な結果です。「有効なSignalが見つからなかった」ことは
    失敗ではなく、事前定義されたSignalがOOSで有効ではなかったことを
    正しく検出できたことを意味します(指示書section 26参照)。
    `research/signal_notes/*.md`に全12 Signalの詳細な数値を記録して
    います。

17. **Market RegimeはBULL/NEUTRALのみが実データで観測され、BEARは
    1件も観測されませんでした。** 2022-2024年のTOPIX Proxyが60日
    トレーリングリターンで-5%を一度も下回らなかったためで、
    `backtest/market_regime.py`の実装自体はBEARを正しく検出できる
    ことを合成データのテスト(`tests/test_backtest_market_regime.py`)
    で確認済みです。実データにBEAR相場が含まれていないだけで、実装の
    不備ではありません。

## Phase 14以降について

Phase 13の完了報告後は停止し、次の指示を待ちます。Strategy tuning・
実運用への自動発注・自動売買・Streamlit UI・Strategy Version 2・
新規Signal発明・Score調整のいずれにも自動で進みません。Forward Test
の日次実行は`.github/workflows/forward_test.yml`により平日21:00 JST
に自動実行されますが、それ以上の機能追加(通知連携・複数銘柄戦略の
追加等)は別途指示がない限り行いません。Phase 11でREJECTとなった
11 Signal、Phase 12でREJECT/REGIME_DEPENDENT_ENSEMBLEとなった
Ensemble候補、Phase 13で発見した条件別仮説のいずれについても、
それを理由にSignal条件・重み・閾値を変更しての採用は行いません。
Phase 13の仮説を実際に検証するには、本Phaseとは完全に独立したOOS
期間での再検証が必要です。

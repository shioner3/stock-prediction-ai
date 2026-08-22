# PHASE 10 REPORT: Frozen Strategy Forward Test Engine

実行日: 2026-08-20

**本レポートは仕様section 24の指定通り、以下10項目のみを報告する。
Forward Testは今日T0が確定した段階であり、実績データはまだ1日分(かつ
未決済トレード0件)しか存在しない。したがって戦略の有効性については
一切結論を出さない。** 最低6か月・可能なら12か月以上の観測期間が必要
であり(仕様section 18)、それまでは`ROBUST`/`REGIME_DEPENDENT`等の
Phase 9で使った分類も、`Historical-consistent`等のPhase 10独自の分類
(仕様section 19)も一切適用しない。

---

## 1. Frozen Strategy Hash

`data/forward_test/manifest.json`の`hashes`フィールドに記録(SHA256、
`features/`・`signals/`・`scoring/`ディレクトリ全体、
`backtest/engine.py`、`backtest/market_regime.py`個別ファイル単位):

| 項目 | Hash |
|---|---|
| features_hash | `20c2748a0cc873e8c3611443ed3590190554b5cb9c832cd19ef19192cdcc35b3` |
| signals_hash | `20a79921c98f2a46a0c1cdca842ab347696cb9cd770fd87a71b987999c0f5df3` |
| scoring_hash | `f3c7510a053cdd7688e300df38691fa82ec21fb161e0ca3e5c0780ca17473872` |
| backtest_engine_hash | `8c7a8fb0222dc395f701931dd630e7be8894b0a4f6150aa176104c3a80f53870` |
| market_regime_hash | `23d037fecb8732be7a9290952b5cef47f61812cf04d857f80ff9057aefa2231b` |

Forward Test実行のたびにこれらを再計算し(`forward_test.manifest.
verify_strategy_hashes_unchanged()`)、1つでも変化していれば
`StrategyHashMismatchError`を送出して即座に停止する仕組みを実装済み
(`tests/test_forward_test_manifest.py`、
`tests/test_pipeline_run_forward_test.py`で検証)。変更が必要になった
場合は仕様section 11の通り、Strategy Version 2として完全に別管理し、
Version 1のForward Testは中断せず継続評価する設計とした。

---

## 2. Frozen Config Hash

`config_hash` = `a9b34ccb6e6e1e1a9d4d1eb554fe3fd8a4c09b2d9d186b2bd0ee3b0d430a2217`

Phase 6.5・Phase 7以来一貫して使用している値と完全一致(`config/settings.yaml`
+ `config/universe_filters.yaml`のsha256結合)。Phase 10専用のパラメータ
(初期資本・1トレードあたりnotional比率等)は、この既存hashに一切影響を
与えない別ファイル(`data/forward_test/manifest.json`自身)に記録する
設計とした(Phase 9で`config/phase9_settings.yaml`を分離した際と同じ
方針)。

---

## 3. Forward Test開始日(T0)

**T0 = 2026-08-20**

`data/forward_test/manifest.json`に不変値として記録。以降のForward Test
実行は全てこのT0を読み込み、`signal_date >= T0`の条件で新規Signal/
Trade判定を行う(T0以前の期間はFeature warmup専用のlookbackとしてのみ
使用し、Forward Test評価対象には一切含めない - 仕様section 3)。

---

## 4. Universe

Phase 6.5〜9と同一の凍結フィルタロジック(`universe/build.py::
apply_static_filters`、`universe/filters.py::check_price_and_liquidity`、
いずれも無変更)を、2026-08-20時点の最新JPX Master
(`data/reference/jpx_master_current.xls`)に適用。

| 項目 | 値 |
|---|---|
| JPX Master候補数 | 4,444 |
| Static Filter通過(Prime+Standard+Growth) | 3,713 |
| Fetch成功 / 部分成功 / 失敗 | 3,622 / 91 / 0 |
| **Final Universe(T0時点)** | **2,780銘柄** |

Universeの銘柄「一覧」自体は凍結せず、フィルタ**ロジック**のみを凍結
している(仕様section 5: 新しい銘柄選定条件は追加していない)。今後
上場・廃止・市場区分変更があれば、同じロジックの下でUniverseは日々
自動的に追随する。

---

## 5. Data Source

- Provider: 既存の`providers/yfinance_provider.py`(無変更、yfinance経由)
- Adjusted/Unadjusted: Phase 1〜9と同一設定を維持(変更なし)
- Lookback: T0からさかのぼり650暦日分を毎回フル取得(Feature最大窓
  SMA200に対する warmup 目的。既存Providerに差分取得機能が無いため、
  日次で全期間を再取得する設計 - 詳細は`pipeline/run_forward_test.py`
  のdocstring参照)
- データ取得失敗銘柄は`FAILED`としてfetch manifestに記録、推定値・
  補完値による穴埋めは一切行っていない

---

## 6. Execution Specification

Phase 4のBacktest仕様を完全に無変更のまま再利用(`backtest/engine.py`
そのものを直接呼び出し):

```
Signal date = t
Entry       = Open[t+1]
Exit        = Close[t+1+HOLD_DAYS-1]   (HOLD_DAYS=5、変更なし)
Cost        = base cost tier(30bps、変更なし)
```

実注文・証券会社API接続・自動発注は一切実装していない(仕様section 22)。
`forward_test/portfolio.py::Position`データクラスに口座・注文ID・API
キー等の実運用フィールドが存在しないことをテストで直接検証
(`test_forward_test_cli_default_directories_are_isolated_from_other_phases`
及び`test_run_forward_test_day_portfolio_has_no_live_order_fields`)。

---

## 7. Paper Portfolio仕様

- 初期資本: 10,000,000円(任意のpaper capital)
- 1トレードあたりnotional: 初期資本の1%(固定・複利なし)
- 同時最大保有数: 上限なし(既存Backtest Engineが銘柄・Signal単位の
  重複抑制のみを行い、ポートフォリオ全体の保有数上限を持たないため、
  それをそのまま踏襲 - 仕様section 6「既存仕様を維持する」)
- Trade Recordは`backtest/engine.py::run_backtest_for_ticker()`
  (無変更)から得られる値をそのまま使用し、金額換算のみPhase 10で新規
  に定義(既存Backtest Engineに元々position sizingモデルが無いため)
- 状態は`data/forward_test/portfolio/portfolio.json`に**追記専用**で
  保存 - 一度記録した`(ticker, signal_name, direction, signal_date)`
  組み合わせは二度と上書きされない(§9のテストで直接検証)

T0時点(初日)では、Entry=Open[T0+1]が未確定のため決済済みポジションは
0件、equityは初期資本のまま10,000,000円。

---

## 8. Signal Logging仕様

`data/forward_test/signals_log/signal_log.jsonl`にJSON Lines形式で
追記専用保存。1行が1 Signal発生(ticker/signal_date/direction/
signal_name/total_score/regime/logged_at)。**Signal発生時点で計算
されたScoreは後から再計算・上書きしない**(仕様section 7の最重要
ルール)。

T0(2026-08-20)当日の実データで、既に4件のSignalが検出された:

| Ticker | Score | Regime |
|---|---:|---|
| 5858 | 24.0 | NEUTRAL |
| 598A | 5.0 | NEUTRAL |
| 6367 | 13.0 | NEUTRAL |
| 7061 | 20.0 | NEUTRAL |

これらは全てNEUTRAL regime。BEAR regimeでのSignal発生は今回は無かった
(規模不足のため`INSUFFICIENT_SAMPLE`として扱う対象にすら至らない、
単なる観測開始日時点のデータ)。**これらSignal 4件についても有効性の
判断は一切行わない。**

---

## 9. Integrity Tests

Phase 10で新規実装したテスト(仕様section 20の10項目に対応):

| # | 項目 | テストファイル |
|---|---|---|
| 1 | frozen config hash | `test_forward_test_manifest.py` |
| 2 | frozen code hash | `test_forward_test_manifest.py` |
| 3 | no future data | `test_pipeline_run_forward_test.py::test_run_forward_test_day_never_fetches_beyond_run_date` |
| 4 | signal immutability | `test_pipeline_run_forward_test.py`(同日再実行での重複なし検証) |
| 5 | daily snapshot reproducibility | `test_pipeline_run_forward_test.py`(同一入力での再実行結果一致検証) |
| 6 | duplicate signal prevention | `test_forward_test_integrity.py`, `test_forward_test_portfolio.py` |
| 7 | position accounting | `test_forward_test_portfolio.py` |
| 8 | virtual execution | `test_pipeline_run_forward_test.py::test_run_forward_test_day_portfolio_has_no_live_order_fields` |
| 9 | missing data handling | `test_forward_test_integrity.py` |
| 10 | historical result isolation | `test_pipeline_run_forward_test.py::test_forward_test_cli_default_directories_are_isolated_from_other_phases` |

T0実データ実行時のIntegrity結果(実運用上の一例):

- Data integrity: 4銘柄(4171, 6197, 6772, 8283)で`is_stale=True`を検出
  (yfinanceの最新データが2026-08-05〜08-19付近で止まっており、
  2026-08-20時点のデータがまだ提供されていない)。生データを直接確認し、
  推定値による穴埋めは行わず、そのまま`STALE`として記録した。監視機構
  が意図通りに機能した実例。
- Trading integrity: `is_clean=True`(重複・不正な価格・不正な日付順序
  いずれも検出されず)

---

## 10. Tests / pytest / ruff / mypy

```
pytest: 587 passed, 2 deselected
ruff: All checks passed
mypy: Success (102 source files)
```

新規モジュール: `forward_test/manifest.py`, `forward_test/portfolio.py`,
`forward_test/integrity.py`, `pipeline/run_forward_test.py`,
`scripts/run_forward_test_day.py`。既存モジュール
(`backtest/engine.py`, `backtest/market_regime.py`,
`pipeline/build_features.py`, `pipeline/build_signals.py`,
`pipeline/build_scores.py`, `pipeline/run_backtest.py`,
`pipeline/universe_ingest.py`, Signal/Score計算群)への変更は一切なし
(全てそのまま再利用)。

---

## 付記: Forward Test Engine Ready 確認

T0(2026-08-20)分の実データを使って上記エンジンをend-to-endで1回実行し、
以下を確認した:

- Universe構築 → データ取得(3,713候補中3,622成功/91部分成功/0失敗)
  → Feature/Signal/Score計算(2,780銘柄) → Backtest再導出まで、
  クラッシュなく完走
- Strategy Hash検証機構が正しく動作(初回作成時にhashを記録)
- Signal Logが追記専用で正しく書き込まれた(4件)
- Paper Portfolioが正しく初期化され、T0時点でequity=10,000,000円
  (決済済みポジション0件、Entry未確定のため妥当)
- Data/Trading Integrityチェックが実データ上の実際の異常
  (4銘柄のstaleデータ)を正しく検出
- ディレクトリ構成(`data/forward_test/{manifest.json, daily/, signals_log/,
  portfolio/, trades/, reports/, raw/, processed/, features/, signals/,
  scores/}`)を仕様section 21通りに生成

**Forward Test Engine Ready。**

今後、`python scripts/run_forward_test_day.py`を営業日ごとに実行する
ことでForward Testを継続できる(現時点では自動的なスケジュール実行は
設定していない - 定期実行が必要な場合はご指示いただければ設定する)。

**最低6か月間はstrategy tuningを行わない。実際のForward Test結果が
蓄積するまでは、戦略の有効性について一切結論を出さない。**

Phase 10実装完了。Phase 11以降(自動売買・Strategy tuning等)には
進まない。

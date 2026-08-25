# Phase V3-7 報告: V1 + V3 Independent Forward Test Integration

本Phaseは指示書どおり、**V3 ML Ranking Forward Observationインフラの構築のみ**
を実施しました。V3の性能に関する結論(「有効」「利益が出る」「実運用可能」等)は
**一切出していません**。V1 Forward Test・V2・V3-1〜V3-6のコード・仕様は
完全に無変更です(`git status`で確認済み、下記「Git Diff範囲」参照)。

---

## 1. V3-5 Frozen仕様(2回のAskUserQuestionで確定した内容)

ユーザーとの2往復の確認により、以下が確定しました:

- **Canonical Targetは設定しない。** 4つのTarget定義(Raw / TOPIX-relative /
  Beta-adjusted Residual / Sector-relative)× 4 Horizon(5/10/15/20d)=
  **16モデルすべてを平等にFreeze・Forward Observation**する。
- 16モデルは `V3-FROZEN-{RAW|TOPIX|BETA|SECTOR}-{5|10|15|20}D` の命名。
- V3-3/V3-4/V3-5のWFO Window別Model(研究用、使い捨て)は一切流用しない。
  今回のFrozen Modelは**T0までの全データで新規に1回だけ学習**した、
  Forward Observation専用の別物。

## 2. Forward Test アーキテクチャ

```
V1 Forward Test (既存、無変更)
  └─ data/forward_test/{raw,processed,manifest.json,signals_log/,...}

V3 Frozen Model Forward Observation (本Phase新規)
  └─ data/forward_test/v3/
       ├─ v3_frozen_models_manifest.json  (16モデルのHash・学習期間等)
       ├─ models/*.txt                     (LightGBM native形式、16個)
       ├─ predictions_log.jsonl            (append-only, 観測日ごと)
       ├─ realized_returns_log.jsonl       (append-only, 満期分のみ)
       └─ daily/<date>.json                (日次サマリ)
```

V3はV1が毎日フェッチする同一OHLCVキャッシュ(`data/forward_test/processed`)を
`V3Config.source_processed_dir`の差し替えのみで再利用し、独自の再フェッチは
行っていません。

## 3. T0の定義

**T0 = 2026-08-20**(V1 Forward TestのT0と同一)。Frozen Model 16個すべて、
2022-01-04〜2026-08-20の全データ(3,085,158行)で1回だけ学習しました。

## 4. V1/V3の状態分離

V1とV3は状態ディレクトリ・ManifestファイルからCLIスクリプトまで完全に分離。
ポートフォリオ・シグナル・戦略判断は一切統合していません(V1は`data/forward_test/`
直下、V3は`data/forward_test/v3/`配下のみ)。GitHub Actionsのジョブ内でも
V3のステップはV1の成否(`SAFE_ABORT`含む)に関わらず独立して実行され
(`if: always() && steps.forward_test.outcome != 'cancelled'`)、どちらの失敗も
他方を止めません。

## 5. Append-only設計

`predictions_log.jsonl` と `realized_returns_log.jsonl` の2ファイル構成。
両方とも `(observation_date, ticker, model_id)` キーで書き込み前に既存キーと
照合し、**既存キーは再書き込みしない**(dedup)。Predictionは一度書いたら
不変、Realized Returnは満期後に別ファイルへ追記されるのみで、既存行の
上書き・削除は一切行いません(`v3/frozen/observation_log.py`)。

## 6. Hash保護

`v3/frozen/manifest.py::verify_frozen_models_unchanged()` が観測実行の
たびに `feature_hash` / `residual_target_hash` (`v3/residual/targets.py` +
`v3/robustness/beta.py`) / `config_hash` / `code_hash` (`v3/`全体) を
**現在のコードから再計算**し、学習時に保存した値と突き合わせます。
不一致時は `FrozenModelHashMismatchError` → exit code 3 で
**SAFE_ABORTと明確に区別して失敗**します(GitHub Actions側も
「V3 FAILURE (exit code 3): FROZEN_MODEL_HASH_MISMATCH」として扱う設計)。

## 7. Leakage保護

`v3/frozen/leakage_check.py::run_observation_shock_checks()` が
observation_date **より後**のOHLCVを`v3/leakage/shock_tests.py`
(V3-1で確立、無変更)のプリミティブでショックし、Prediction結果が
ショック前後でビット一致することを検証します。統合テストで実データ形状の
合成データに対し3種のショックすべてPASSを確認済み。

## 8. データ整合性保護

V1既存の `forward_test/integrity.py::check_data_integrity()` を無変更で
再利用し、Ticker毎のStale判定 → `stale_fraction > 0.5` で
`SAFE_ABORT[STALE_THRESHOLD_EXCEEDED]`。V1のフェッチマニフェスト
(`data/forward_test/_fetch_manifest.json`)が存在しない/空の場合は
`SAFE_ABORT[MARKET_DATA_UNAVAILABLE]`。

**実データ実行で判明した事実(バグではなく観測結果として報告)**:
T0=2026-08-20のObservation実行時、TOPIX Proxy(`1306.T`)自身のClose値が
V1の`data/forward_test/processed`キャッシュ内でT0の行のみ`NaN`でした
(432/433行は正常、T0のみ欠損)。これにより`backtest/market_regime.py::
compute_market_regime()`の既存ロジック(warmup/NaN区間を`None`にマスク、
無変更)が正しく動作し、T0の全44,448 Prediction行の`regime`フィールドが
`"None"`になっています。**Prediction・Rank・Percentile自体はFeature
Panelから独立に計算されるため無影響**であり、この欠損はメタデータ
(regime表示)のみに限定されます。既に書き込み済みのPrediction行は
Append-only設計により**再計算・上書きしていません**。今後V1の日次
フェッチが進めばTOPIX Closeは通常埋まる見込みですが、これは観測事実の
記録に留め、本Phaseでは修正を行っていません。

## 9. Idempotency(冪等性)

同一観測日を2回実行しても新規行は0件(キー重複を検出しスキップ)。
`tests/test_v3_frozen_observation_log.py`のテスト、および実データ
Observation実行時に`append_prediction_entries`を2回連続で呼び出し
(1回目400件→2回目0件、25銘柄Dry Run時点)で確認済みです。

## 10. GitHub Actions設定

既存 `.github/workflows/forward_test.yml` を**拡張**(新規ワークフローは
作成せず)。V1の既存ステップの直後、Commitステップの直前に
`scripts/run_v3_frozen_observation_day.py`を実行する4ステップを追加。
Exit code規約: 0=成功, 2=SAFE_ABORT(想定内, 非失敗), 3=
FROZEN_MODEL_HASH_MISMATCH(失敗)。Commitステップの`git add`対象に
`data/forward_test/v3/`ディレクトリ全体を追加(個別ファイル指定だと
初日に存在しないファイルへの`git add`がpathspecエラーになるため、
V1の`reports/`と同じ`mkdir -p`+ディレクトリ単位addパターンを踏襲)。
YAML構文は`python -c "import yaml; yaml.safe_load(...)"`で検証済み。

**未実施**: 本ローカル環境に`gh` CLIが存在しないため
(`gh: command not found`)、GitHub Actions上での実行確認(仕様書
セクション32の要求)は本Phase内では完了していません。このブランチを
pushした後、GitHub の Actions タブから `Forward Test daily run` ワーク
フローを手動トリガー(workflow_dispatch)するか、次回の定期実行
(平日12:00 UTC / 21:00 JST)を待つ必要があります。Commit/Pushは
ユーザーの明示的な許可を得てから行う方針のため、本Phaseではここで停止し、
実行はユーザー確認後とします。

## 11. Dry Run結果

- **学習パイプライン**: 実データ25銘柄限定でscratchディレクトリに対し
  16モデル学習 → Hash検証(config/feature一致、dataset/code不一致は
  ticker制限による既知の差分)→ 20秒で完了。
- **Observationパイプライン**: 同じ25銘柄・実データでObservation構築
  (400件 = 25銘柄×16モデル)、2回連続実行で冪等性確認、Realized Return
  検出0件(最新日=満期データなし、想定通り)。

## 12. 実データ本番実行結果

### 学習(`scripts/train_v3_frozen_models.py`, ticker制限なし)

- Full Universe dataset: **3,085,158行**、70列、期間 2022-01-04〜2026-08-20
- Hash検証: `config_hash_match=True`, `feature_hash_match=True`,
  `dataset_hash_match=True`(V3-3/V3-4/V3-5の凍結仕様と完全一致)
- 16モデル全て学習成功(学習行数はTarget/Horizonにより2,906,809〜
  3,070,751行の範囲、NaN・implausible値除外後)
- 初回実行は3M行×約79列のワイドDataFrameに対する`prepare_training_set`
  の内部Consolidate処理で**MemoryError**が発生(1モデル目で停止、
  0モデル完了)。原因はメモリ効率の問題であり統計手法・Feature・Target・
  Hyperparameterとは無関係と判断し、`v3/frozen/train.py`にて
  (a) 各モデル学習前に必要列のみへ絞り込み、(b) 学習完了後は
  列名のみ保持した軽量`TrainingSet`のみ返却・実データは即解放、
  `scripts/train_v3_frozen_models.py`にて(c) 拡張後に不要となった
  raw datasetを`del`+`gc.collect()`、の3点で修正。再実行し
  **16/16モデル成功**、`v3_frozen_models_manifest.json`保存完了。

### Observation(`scripts/run_v3_frozen_observation_day.py --run-date 2026-08-20`)

- Universe size: **2,778銘柄**
- Prediction entries built: **44,448**(= 2,778銘柄 × 16モデル)
- Prediction entries new: 44,448(初回実行のため全件新規)
- Realized return entries new: **0**(T0当日のためHorizon未満期、想定通り)
- regime: 全件`"None"`(§8参照、TOPIX Proxy当日Close欠損による既存ロジックの
  正しい挙動)

**運用上の注意**: Observation実行を2重起動してしまうミスが発生しましたが
(誤った変数展開によるログリダイレクト先の混乱が原因)、両プロセスとも
書き込みフェーズに到達する前に検知し、片方を`taskkill`で即座に終了させ
ました。書き込み済みファイルへの影響がないことを`ls`で確認済みです。

## 13. テスト件数

- V3-7専用テスト: `tests/test_v3_frozen_manifest.py`(5件)、
  `tests/test_v3_frozen_observation_log.py`(5件)、
  `tests/test_v3_frozen_realize_returns.py`(4件)、
  `tests/test_v3_frozen_integration.py`(1件、End-to-End)= **15件、全PASS**
- プロジェクト全体テストスイート: **1078 passed, 2 deselected**
  (845.55秒、V3-7追加によるregressionなし)

## 14. ruff結果

V3-7対象ファイル(`v3/frozen/`, `scripts/train_v3_frozen_models.py`,
`scripts/run_v3_frozen_observation_day.py`, `tests/test_v3_frozen_*.py`)
に対し `ruff check` — **All checks passed!**

## 15. mypy結果

`v3/frozen/`, `scripts/train_v3_frozen_models.py`,
`scripts/run_v3_frozen_observation_day.py` に対し `mypy` —
**Success: no issues found in 10 source files**

## 16. Git Diff範囲

```
modified:   .github/workflows/forward_test.yml

untracked:  v3/frozen/ (新規パッケージ)
            scripts/train_v3_frozen_models.py
            scripts/run_v3_frozen_observation_day.py
            tests/test_v3_frozen_manifest.py
            tests/test_v3_frozen_observation_log.py
            tests/test_v3_frozen_realize_returns.py
            tests/test_v3_frozen_integration.py
            data/forward_test/v3/ (Frozen Model 16個 + Manifest +
                                    predictions_log.jsonl、約29MB)
```

V1(`forward_test/`, `pipeline/run_forward_test.py`等)・V2・
V3-1〜V3-6(`v3/dataset.py`, `v3/models/`, `v3/robustness/`,
`v3/residual/`等)への変更は**ゼロ**です。

## 17. 現在のForward Observation件数

- Prediction entries: **44,448件**(観測日1日分: T0=2026-08-20)
- Realized return entries: **0件**

## 18. 満期済み(Resolved)観測件数(Horizon別)

| Horizon | Resolved件数 |
|---|---|
| 5d  | 0 |
| 10d | 0 |
| 15d | 0 |
| 20d | 0 |

T0当日のみの単発実行のため、いずれのHorizonも未満期です。今後V1の日次
フェッチが進み、GitHub Actions経由での日次Observationが継続することで、
5d Horizonから順次満期していく見込みです(本Phaseではこの先の実行や
判断は一切行いません)。

## 19. 現在の状況

- 16 Frozen Model学習・永続化: **完了**
- Forward Observationインフラ(Predict/Log/Realize/Idempotency/Hash/
  Leakage/Data Integrity): **実装・テスト完了**
- 実データでのDry Run・本番学習・本番Observation(T0初日分): **完了**
- GitHub Actions拡張の実装・YAML検証: **完了**
- GitHub Actions上での実行確認(CI上で最低1回動作させる): **未完了**
  (`gh` CLI不在のため。Push+手動トリガーまたは次回定期実行が必要)
- Commit / Push: **未実施**(ユーザーの明示的許可待ち)
- `research/phase_v3_7_report.md`(本ファイル): **完了**

**本Phaseでは、V3の性能・有効性について一切の結論を出していません。**
これは意図的な設計であり、性能評価は将来、別途事前登録される
Independent OOS Phaseの役割です。

---

以上でPhase V3-7を停止します。V3-8、Tuning、V1/V2統合、実運用、
Streamlit UI、Broker API連携などへは一切進みません。

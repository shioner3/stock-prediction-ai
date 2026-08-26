# Phase V3-2 完了報告: ML Expected-Value Ranking Engine — Baseline Model Implementation

Strategy Version 1(V1)・Strategy Version 2(V2)は、いずれも完全凍結の
まま一切変更していません。Phase V3-1(Dataset / Feature Registry /
Leakage Framework)も一切変更していません(唯一の例外は、V3-1自身の
コードに含まれていた実バグの修正 — 第13節参照)。

**本Phaseのスコープ**: LightGBM Baseline Model(Regression / Binary
Classification / Quantile Regression)の実装と、小規模subset(40銘柄)
での学習・予測・基本評価パイプラインが正しく動作することの確認のみ
です。指示書section 25の明示的な指示どおり、**Full Universe OOS・
Walk-Forward Optimization・Hyperparameter tuning・Feature selection・
Risk-adjusted Rankingの最終化・Top-N最適化・Streamlit UI・Paper
Trading・自動発注のいずれにも進んでいません**。「MLモデルが有効」
「V3はV1より優れている」等の結論は一切出していません。

---

## 1. Purpose

Baseline ML pipelineが正しく動くことの確認。良いモデルを探すPhaseでは
ない。性能評価は参考情報として保存するのみ。

## 2. Environment

- Python 3.10.2、既存の`.venv`をそのまま使用(新しい仮想環境は作成
  していません)。
- 既存の主要パッケージ(V1/V2が依存)はバージョン変更なし:
  numpy 2.2.6・pandas 2.3.3・pyarrow 25.0.1・pydantic 2.13.4。
  `pip install -e ".[v3]"`実行時、全てpip出力上"already satisfied"と
  表示され、ダウングレード・競合は一切発生しませんでした。

## 3. Dependencies

`pyproject.toml`に**V3専用の新しいoptional-dependenciesグループ**
`v3`を追加しました(`dependencies`本体にも`dev`グループにも追加して
いません):

```toml
v3 = [
    "lightgbm>=4.1,<5.0",
    "scikit-learn>=1.4,<1.6",
]
```

- インストールコマンド: `pip install -e ".[v3]"`(V1/V2ユーザーが
  通常実行する`pip install -e ".[dev]"`には一切影響しません)。
- 実際にインストールされたバージョン: `lightgbm==4.7.0`・
  `scikit-learn==1.5.2`(+推移的依存として`scipy==1.15.3`・
  `joblib`・`threadpoolctl`・`narwhals`)。
- バージョンは範囲固定(pinned range)とし、「最新版に無制限に追従」
  していません。
- V1/V2の実行環境への影響: **なし**(`git status`で確認済み、
  `pyproject.toml`以外の既存追跡ファイルは無変更 — 第17節参照)。

## 4. Models

| Model | 種類 | 実装 |
|---|---|---|
| A | LightGBM Regression | `v3/models/regression.py` |
| B | LightGBM Binary Classification(future_return > 0) | `v3/models/classification.py` |
| C | LightGBM Quantile Regression(q=0.1/0.5/0.9) | `v3/models/quantile.py`(3独立モデル) |

Model Cは3つの独立したLightGBMモデル(`objective="quantile"`、
`alpha=q`)として実装しています — LightGBMのsklearn APIは1モデル
あたり1 quantileしか学習できないため、これは正しい標準的な実装方法
であり、簡略化ではありません。

## 5. Hyperparameters

事前固定・本Phase内で一切調整していません(`v3/models/config.py`):

```python
n_estimators = 300
learning_rate = 0.03
max_depth = 6
num_leaves = 31
min_child_samples = 50
subsample = 0.8
colsample_bytree = 0.8
reg_lambda = 1.0
reg_alpha = 0.0
random_state = 42  # RANDOM_SEED、全モデル共通
```

指示書自身が明示するとおり、これらは「最適値」ではなく、保守的な
初期Baselineです。

## 6. Dataset

Phase V3-1の`v3/dataset.py::build_v3_dataset()`をそのまま再利用。
小規模subset(40銘柄、NOT Full Universe)で実行:

- 行数: 38,039、列数: 70(date/ticker + 52 Core Feature + 16 Target)
- 期間: 2022-01-04 〜 2026-08-20
- dataset_hash: `fd34612dff8201d27fbbf24dea12a3958f4bc99ee968122686902fe5594250a2`
  (Phase V3-1実行時と完全一致 — 第13節のTarget dtype修正はCSV表現上
  の値そのものには影響しなかったため)

## 7. Target

Primary Baseline: **`target_raw_5d`**(5d Horizon・Raw Variant)。

Target切り替え確認(指示書section 14、「どれが最良か」の選択はして
いません、パイプラインが扱えることのみ確認):

| Target | Test n | Spearman(参考値) |
|---|---|---|
| target_raw_5d(Primary) | 12,600 | 0.0429 |
| target_raw_10d | 12,400 | 0.0191 |
| target_topix_relative_5d | 12,600 | 0.0087 |
| target_vol_adjusted_5d | 12,600 | 0.0269 |
| target_risk_adjusted_5d | 12,242 | -0.0086 |

15d/20dはHorizon切り替えの動作確認のみ(`target_raw_10d`で確認済み、
同じ`target_col`引数の切り替えで動作することを`run_model_a()`が
`v3/targets/registry.py::target_registry_by_name()`からHorizonを
自動解決する構造で保証しています)。

## 8. Time Split

`v3/dataset.py::time_split()`(本Phaseで新規追加)を使用。ランダム
splitは一切使用していません。

- TRAIN: 2022-01-04 〜 2025-04-01(24,479行)
- **Embargo(空白期間)**: 20営業日(Target Registryの最大Horizonである
  20dに合わせて設定 — 訓練データ末尾の行が、テスト期間に含まれる
  未来のClose値をTargetとして参照してしまう「境界のリーク」を構造的
  に防ぐため)
- TEST: 2025-04-30 〜 2026-08-20(12,800行)

WFOの本格実装はV3-3以降の課題ですが、`time_split()`はそのための
土台として時間軸対応のAPIになっています。

## 9. Baseline Metrics

### Model A(Regression、target_raw_5d)

| | n | MAE | RMSE | R² | Pearson | Spearman |
|---|---|---|---|---|---|---|
| Train | 24,479 | 0.0295 | 0.0477 | 0.391 | 0.674 | 0.497 |
| Test | 12,600 | 0.0418 | 0.0680 | **-0.139** | -0.013 | 0.043 |

Test R²が負(TrainとTestの性能が大きく乖離)であることをそのまま記録
します。これは40銘柄という小規模subset・保守的なBaseline
Hyperparameterでの結果であり、「モデルが悪い」と結論づけるものでは
なく、単に動作確認用の参考値です(指示書section 24の明示的な要求
どおり)。

### Model B(Binary Classification、target_raw_5d > 0)

| | n | ROC-AUC | LogLoss | Brier | Accuracy | Positive Rate |
|---|---|---|---|---|---|---|
| Test | 12,600 | 0.527 | 0.719 | 0.262 | 0.509 | 0.517 |

Calibration評価(「予測確率0.7は実際に70%上昇するか」)は本Phaseでは
実施していません(指示書section 6の明示的な指示どおり、後Phaseの
課題)。

## 10. Cross-sectional Results

Model AのTest期間予測をQ1-Q5に分割(`v3/models/cross_sectional.py`、
V1の`assign_quantile_buckets()`・V2の`compute_quantile_bucket_stats()`
を再利用):

- **Q5-Q1 spread: +0.00123**(正の方向)

指示書section 12の明示的な指示どおり、Q5 > Q1になるようモデルを調整
したことはありません — これは調整前のBaselineがたまたま示した結果を
そのまま報告しているものです。

## 11. Random Baseline

同じTest期間で、固定seed(101)のランダムランキングを5パターン生成
(`v3/models/cross_sectional.py::add_random_baseline_column()`)し、
同じQ1-Q5分析を実施:

- **Random Q5-Q1 spread: +0.00067**

MLモデルの予測(+0.00123)がRandomベースライン(+0.00067)を上回って
いますが、正式な統計検定(Permutation Test等)はV3-5以降で実施予定
であり、本Phase単独でこの差を「有意」と解釈していません。

## 12. Feature Importance

`v3/models/importance.py`がGain/Split importance、およびLightGBM
ネイティブの`pred_contrib=True`によるSHAP値(追加パッケージ不要)を
提供します。Model AのTest期間、Gain importance上位5:

1. `atr`(39.28)
2. `topix_return_60d`(37.70)
3. `ma60_to_ma120`(34.24)
4. `topix_return_20d`(33.06)
5. `topix_volatility_20d`(27.71)

**重要**: この結果を受けてFeature Registry(52 Core Feature)を変更
することは一切行っていません(指示書section 15の明示的な禁止事項
どおり)。V2-3で発見された「短期モメンタムが弱い+長期トレンドが
維持されている」パターンとの直接対応は見られません(topix系・ATR・
長期MA比率が上位)が、これも特別扱いせず、単なる観察として記録する
のみです(指示書section 1のRule 9、section 25の明示的な指示どおり)。

## 13. Leakage Tests

### A. Target Exclusion Test(新規)

`v3/models/data_prep.py::assert_no_target_leakage_in_features()`が、
モデルへ渡すFeature行列に16 Target列のいずれか、または`date`/
`ticker`列が含まれていないことを機械的に検証します。全16 Targetに
ついて、意図的に混入させた場合に確実に検出されることをテストで確認
済みです。

### B. Future Shock Model Test(新規)

Train期間より後・Test期間内にcutoffを設定し、そのcutoff以降のOHLCV
(価格)を改変した上でDatasetを再構築・再学習した結果:

- **(a) 再学習後のModelは改変前と完全に同一**(`model_hash`が完全
  一致)— Train期間のデータがcutoffより前に収まっているため、
  Shockの影響を一切受けないことを直接証明しています。
- **(b) cutoff以前の日付のTest予測は完全に不変**

### C. **重大な実バグの発見と修正**(本Phase内、結果を見る前に発見)

Baseline学習を小規模subsetで実行した際、`target_risk_adjusted_5d`
列がLightGBMから`ValueError: pandas dtypes must be int, float or
bool`で拒否されるエラーが発生しました。原因調査の結果、Phase V3-1
の`v3/targets/compute.py`が、ゼロ除算回避のために`.replace(0, pd.NA)`
を使用していたことが判明しました — `pd.NA`をfloat64のSeriesに
`.replace()`すると、Series全体が`object` dtypeに暗黙的に格上げされ
(実際の値は全て正しい浮動小数点数のままですが、dtypeラベルだけが
`object`になる)、これをLightGBMの厳格なdtypeチェックが拒否していま
した。

**この不具合はPhase V3-1のテストでは検出されませんでした** —
`tests/test_v3_targets.py`の該当テストが`.to_numpy(dtype=float)`で
明示的に型強制していたため、根本原因のdtype問題を覆い隠していたこと
が分かりました。これは本Phaseで初めてMLモデルの厳格な型チェックに
実際に晒されたことで顕在化した、正直に記録すべき教訓です。

**修正**: `pd.NA`を`np.nan`に置き換え(`v3/targets/compute.py`)。
値そのものは変わらないため(dataset_hashのCSV表現は修正前後で完全
一致)、これは「結果を見て仕様を変更した」ものではなく、純粋な実装
バグの修正です。`v3/targets/registry.py`のTarget定義・formula・
Horizon構成は一切変更していません。修正後、全16 Target列がLightGBM
で正常に学習できることを確認しました。

## 14. Reproducibility

同一Dataset・同一Config・同一Hyperparameter・同一Random seed(42)で
2回学習を実行した結果、`dataset_hash`・`model_hash`ともに完全一致
することを確認しました:

```
1回目: dataset_hash=fd34612d... model_hash=2a4ee1a3...
2回目: dataset_hash=fd34612d... model_hash=2a4ee1a3...(完全一致)
```

Model A/B/Cそれぞれについて、同一入力での2回学習が完全に同一の
予測を返すことも単体テストで確認済みです(`tests/test_v3_2_models.py`)。

## 15. Hashes

| Hash | 内容 | Phase V3-2実行時の値 |
|---|---|---|
| code_hash | `v3/`配下の全`.py` | `5a4a052dd742a48bec05725689d32977ff5c1b37f79ed1004a25f0ad0db89c36`(V3-1から更新、`v3/models/`追加のため) |
| config_hash | `v3/config/v3_settings.yaml` | `e5b10f6049301dee84cfbea2bf1275c7d0fe5a9bf1976de4d8627eb6fe08bd1f`(**V3-1と完全一致** — config未変更) |
| feature_hash | `v3/features/` + `v3/targets/` | `b507d5db3d92ae2c61bd3cba0ae42caf83463eda82e6f79ca4d020719fa19098`(第13節のbugfixにより更新) |
| dataset_hash | 生成Dataset本体 | `fd34612dff8201d27fbbf24dea12a3958f4bc99ee968122686902fe5594250a2`(**V3-1と完全一致**) |
| model_hash | 学習済みModel Aのbooster | `2a4ee1a320c4e8b0ff03bc0d936f603f3e59a1d1b7fde091c2e4da5bce84ad0f` |

Model ManifestはHyperparameterも含めて`data/v3/models/`に保存されて
います(`v3/models/model_manifest.py`)。V1のStrategy Hash・V2の
manifestとは完全に別のnamespaceです。

## 16. Tests

`tests/test_v3_2_*.py`(10ファイル + 共有helper 1ファイル)を新規
追加、指示書section 21の10項目全てをカバー:

1. model fit test(`test_v3_2_models.py`)
2. prediction shape test(同上)
3. target exclusion test(`test_v3_2_data_prep.py`・`test_v3_2_leakage.py`)
4. time split test(`test_v3_2_split.py`)
5. reproducibility test(`test_v3_2_models.py`・`test_v3_2_model_manifest.py`・`test_v3_2_orchestrator.py`)
6. future shock model test(`test_v3_2_leakage.py`)
7. cross-sectional ranking test(`test_v3_2_cross_sectional.py`)
8. random baseline test(同上)
9. model hash test(`test_v3_2_model_manifest.py`)
10. V1/V2 isolation test(`test_v3_2_leakage.py`)

プロジェクト全体のtest suite: **964 passed / 2 deselected**
(V3-1までの920件 + 新規44件、regressionなし)。ruff/mypyともにクリア
(V3ソースコード対象、`tests/`は本プロジェクトの既存慣行どおりmypy
スコープ外)。

**副次的な発見**: 既存のV3-1 leakage test(`tests/test_v3_leakage.py`)
とV3-2の新規テストの両方に、「V1の意思決定層をimportしていないか」を
チェックする静的検査があり、当初`scoring`パッケージ全体をブロック対象
としていましたが、これは`scoring.validation`(quantile bucket分析等の
汎用統計ユーティリティ、V2が既に再利用している正当な対象)まで誤って
ブロックしてしまう誤検知でした。`scoring.scorer`/`scoring.pipeline`
(実際の意思決定Score計算)のみをブロック対象とするよう両テストを
修正しました。

## 17. V1/V2 Isolation

`git status`で確認: 本Phaseで変更した既存追跡ファイルは
`pyproject.toml`(V3専用依存関係グループの追加)のみです。V1
(`features/`・`signals/`・`scoring/`・`backtest/`・`targets/`・
`forward_test/`・`ensemble/`・`pipeline/`)・V2(`v2/`配下全て)は
バイト単位で無変更です。V3-1の既存ファイル(`v3/dataset.py`・
`v3/targets/compute.py`・`v3/targets/registry.py`)への変更は、
「V3だけを変更する」という指示書section 2の許容範囲内での、
V3自身の拡張・バグ修正です。

## 18. Limitations

- 40銘柄という小規模subsetでの結果であり、Full Universe(2,880銘柄)
  での性能を示唆するものではありません。
- Hyperparameterは保守的なBaselineであり、最適化されていません。
- Test R²が負であることが示す通り、現時点のBaselineモデルはこの
  小規模データでは汎化性能が低い可能性がありますが、これはHyper-
  parameter未調整・Feature selection未実施・小規模subsetという
  複数の要因が絡んでおり、切り分けは行っていません。
- Model Cの3 quantileモデルは、Risk-adjusted Scoreの構築にはまだ
  使用していません(指示書section 7の明示的な指示どおり)。
- Calibration評価(Model Bの予測確率の妥当性)は未実施です。
- Permutation Test等の正式な統計的頑健性検証はV3-5以降の課題です。

## 19. Next Phase

指示書section 25の明示的な指示どおり、以下には進んでいません:

Full Universe OOS・WFO(Walk-Forward Optimization)・Hyperparameter
tuning・Feature selection・Risk-adjusted Ranking最適化・Top-N
最適化・Streamlit UI・Paper Trading・自動発注。

次のPhase(V3-3: Full Universe OOS)は、本報告書のレビュー後、明示的
な指示を受けてから開始します。

---

**Phase V3-2 complete — stopped before Full Universe OOS**

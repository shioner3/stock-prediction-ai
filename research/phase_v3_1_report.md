# Phase V3-1 完了報告: ML Expected-Value Ranking Engine — Dataset / Feature Registry / Leakage Framework

Strategy Version 1(V1: 既存12 Signal・Score・Backtest・Walk Forward
Validation・Phase 10 Forward Test Engine)・Strategy Version 2(V2:
`v2/`パッケージのRanking Score・Phase V2-2 Full Universe OOS
Validation・Phase V2-3 Causal Decomposition)は、いずれも完全凍結の
まま一切変更していません。本Phaseは、V1・V2とは完全に独立した第3の
研究系統(V3: ML Expected-Value Ranking Engine)の**最初のステップ
のみ**です。

**本Phaseのスコープ**: Dataset構築・Feature Registry・Target
Registry・Leakageフレームワークの実装と、小規模subset(40銘柄)での
Dataset生成検証のみです。指示書section 38の明示的な指示どおり、
**MLモデルの学習・Full Universe OOS検証・Hyperparameter tuning・
Risk-adjusted Rankingの最終化・Streamlit UI・Paper Trading・自動発注
のいずれにも進んでいません**。ML性能についての結論はここでは一切
出していません。

---

## 1. Executive Summary

Phase V3-1のスコープ(指示書section 36の完了条件)を全て満たしました:

- V3 package(`v3/`)・V3独自config(`v3/config/`)・Feature Registry
  (52 Core + 3 Conditional = 55エントリ)・Target Registry(4 Horizon
  × 4 Variant = 16エントリ)・Full-Universe対応のDataset生成コード
  (`v3/dataset.py`)・Leakageテスト(静的AST検査 + Future Shock Test
  4種)・Dataset hash・V3独自hash(config/code/feature/dataset/
  model)・Unit tests(52件、全てpass)・V1/V2 non-modification check
  ・README更新・本報告書を全て作成しました。
- 小規模subset(40銘柄、2022-01-04〜2026-08-20、38,039行×70列)で
  実際にDatasetを生成し、再現性(同一条件での再実行で
  dataset_hashが完全一致)を確認しました。
- Leakageテストは全て合格しました。Future Shock Test(価格・指数・
  出来高・ランダム摂動の4種)を実際のDataset生成パイプライン全体
  (V1 Feature計算 → V3派生Feature → Cross-sectional Feature →
  Market Context)に対して実行し、cutoff日以前のFeature値が
  一切変化しないことを確認しています。
- V1・V2は`git status`で完全無変更(`.gitignore`への`data/v3/`追記と
  README.mdへのV3節追記のみ、コード変更ゼロ)を確認しました。
- **Full Universe実行・ML学習には進んでいません**。次のステップ
  (V3-2: Baseline ML)を開始する前に、本報告書のレビューを受ける
  想定です。

## 2. Repository構造確認(指示書section 35 STEP 1-5)

- リポジトリはV1(`features/`・`signals/`・`scoring/`・`backtest/`・
  `targets/`・`forward_test/`・`ensemble/`・`pipeline/`)・V2
  (`v2/`、Phase V2-1〜V2-3実装済み)が既に存在するブランチ
  (`add-v2-ranking-engine`)を起点に、新しいブランチ
  `add-v3-ml-ranking-engine`を作成して開発しました。V2が未マージの
  ため、`origin/main`を起点にすると`v2/`が存在せず「V2を変更しない」
  ことを`git status`で具体的に検証できなくなるためです。
- V1/V2の関連ファイルは本Phase冒頭で確認済み(`features/pipeline.py`
  ・`targets/forward_returns.py`・`v2/manifest.py`・
  `v2/ranking/cross_sectional.py`等)。
- 利用可能な価格データ期間: **2022-01-04 〜 2026-08-20**(V1/V2と同一
  のFull Universeキャッシュ、`data/phase7/`)。
- Universe: **2,880銘柄**(Prime/Standard/Growth、V1/V2と同一の
  Universe filter)。

## 3. V3 Package構造

```
v3/
  __init__.py              # 独立性の原則を記述
  config/
    loader.py               # V3Config(独立Pydanticツリー)
    v3_settings.yaml
  features/
    registry.py              # FeatureSpec + FEATURE_REGISTRY(55件)
    price_features.py         # V1 utility関数経由の派生Feature
    cross_sectional.py        # V2のpercentile_rank_by_day()を再利用
    market_context.py         # TOPIX指標 + market breadth(新規)
    sector.py                 # 業種別相対力(条件付き、オプトイン)
  targets/
    registry.py               # TargetSpec + TARGET_REGISTRY(16件)
    compute.py                 # Raw/TOPIX-relative/Vol-adjusted/Risk-adjusted
  leakage/
    availability_check.py      # 静的AST検査
    shock_tests.py               # Future Shock Testヘルパー
  dataset.py                # Dataset構築のオーケストレーション
  hash.py                    # V3独自hash namespace
scripts/
  build_v3_dataset.py        # 小規模subset生成CLI
```

## 4. Feature Registry(spec section 5/6)

52 Core + 3 Conditional(業種、オプトイン)= 55エントリ。カテゴリ内訳:

| Category | 件数 | 主なソース |
|---|---|---|
| momentum | 7 | V1再利用(6) + V3新規(1: 120d) |
| moving_average | 8 | V1再利用(2) + V3新規(6) |
| volatility | 6 | V1再利用(5) + V3新規(1: volatility_change) |
| oscillator | 3 | V1再利用(1: RSI14) + V3新規(2: RSI5/20) |
| volume | 5 | V1再利用(3) + V3新規(2: turnover/turnover_ratio) |
| breakout_drawdown | 6 | V1再利用(2: 20d) + V3新規(4: 60d/120d) |
| cross_sectional | 6 | V2の`percentile_rank_by_day()`を再利用 |
| market_context | 9 | V1のTOPIX指標再利用(8) + V3新規(1: market_breadth系のうち2つ) |
| sector(条件付き) | 3 | V3新規、`jpx_master_current.xls`ローカルキャッシュ依存 |

全エントリについて `name/category/formula/required_history/
availability/leakage_risk/source` を記録しています
(`v3/features/registry.py`)。V1が既に計算済みの列は`v1_reuse`として
importのみで再利用し、再実装していません。

## 5. Target Registry(spec section 3/4)

4 Horizon(5/10/15/20d) × 4 Variant = 16列:

- **Raw**: `Close[t+h]/Close[t] - 1`(V1の`compute_forward_returns()`
  を無変更で再利用)
- **TOPIX-relative**: Raw(stock) - Raw(TOPIX Proxy)、日付ベースの
  alignment(`features/relative_strength.py`と同じ規律)
- **Volatility-adjusted**: Raw / `volatility_20d[t]`(t時点で既知の
  リスク推定量で正規化、ex-ante)
- **Risk-adjusted**: Raw / |MAE_h|(保有期間中の実現最大逆行幅で
  正規化、ex-post。V1の`compute_mfe_mae()`を無変更で再利用)

**どのTarget/Variantを最終採用するかは本Phaseでは決定していません**
(指示書section 3の明示的な指示どおり、事前固定した評価基準で将来
Phaseにて決定)。

## 6. Leakageフレームワーク(spec section 6/23)

### A. 機械的AVAILABLE_AT<=t検査

`v3/leakage/availability_check.py`が`v3/features/*.py`の全ソースを
静的(AST)に走査し、(a) 負の引数を持つ`.shift()`呼び出し(未来読み取り)
と、(b) `targets.forward_returns`のimport(V1で唯一未来を読むことが
許されたモジュール)の2種類を検出します。実行結果: **findings=0**。

### B. Future Shock Test(4種、実際のDataset生成パイプライン全体に対して実行)

小規模subset(6銘柄の合成データ)に対し、cutoff日以降のOHLCVを以下の
4通りに改変し、Dataset全体を再構築した上で、cutoff日以前の全Core
Feature列が完全に不変であることを確認しました(`tests/test_v3_leakage.py`):

- **A. Future price shock**: Close/High/Low/Open を5倍
- **B. Future index shock**: TOPIX Proxy自体のClose/High/Low/Openを5倍
- **C. Future volume shock**: Volumeを10倍
- **D. Random future perturbation**: 独立したランダムウォークで置換

4種全てPASS。Cross-sectional Feature(他銘柄の値に依存)・Market
Context Feature(TOPIX/breadthに依存)も含めた**Dataset全体**に対して
検証しており、単一Feature関数の分離テストだけでなく、Universeスタック
後のパイプライン全体でのLeakage不在を確認しています。

### C. V1非依存性の静的検査

`v3/`配下の全ソースが、V1の意思決定層(`signals/`・`scoring/`・
`backtest/`・`forward_test/`・`ensemble/`)を一切importしていないこと
をAST検査で確認しました(V3が読んでよいのは`features/`・
`targets/forward_returns.py`・`universe/`という「純粋な計算層」のみ
- V2が既に確立した同じ再利用パターンです)。

## 7. Hash Integrity(spec section 32/33)

`v3/hash.py`が、V1のStrategy Hash・V2のmanifestとは完全に独立した
hash namespaceを提供します:

| Hash | 対象 | 小規模subset実行時の値(先頭16桁) |
|---|---|---|
| code_hash | `v3/`配下の全`.py` | `f939b2608782308b` |
| config_hash | `v3/config/v3_settings.yaml` | `e5b10f6049301dee` |
| feature_hash | `v3/features/` + `v3/targets/`のみ | `1331856c73d08a8a` |
| dataset_hash | 生成したDataset本体 | `fd34612dff8201d2` |
| model_hash | (Phase V3-1では常にNone) | - |

同一条件での再実行(`scripts/build_v3_dataset.py --limit-tickers 40`
を2回実行)で、`dataset_hash`を含む全hashが完全に再現することを確認
しました(`tests/test_v3_hash.py`でも決定性を単体検証済み)。

## 8. 小規模subset Dataset生成結果(spec section 35.9)

```
STEP 1: Repository confirm
  tracked changes to V1/V2 files (should be 0): 0

STEP 2: Mechanical Feature leakage check (v3/features/*.py, AST scan)
  findings: 0

STEP 3: building small-subset dataset (40 tickers, NOT Full Universe)
  dataset rows: 38039
  dataset columns: 70
  date range: 2022-01-04 .. 2026-08-20
```

Full Universe(2,880銘柄)ではなく**40銘柄のみ**で実行しました
(指示書section 38の明示的な停止指示どおり)。Dataset列数70 = 
date/ticker(2) + Core Feature(52) + Target(16)。

## 9. Unit Tests

`tests/test_v3_*.py`(10ファイル、52 test)を新規追加。カバー範囲:

- Feature Registry / Target Registryの整合性(重複なし・必須field
  充足・Core/Conditionalの排他性)
- 各Feature計算関数(`price_features.py`・`cross_sectional.py`・
  `market_context.py`・`sector.py`)の単体テスト(境界値・入力
  非破壊・決定性)
- Target計算(`compute.py`)がV1の`compute_forward_returns()`/
  `compute_mfe_mae()`と一致すること
- Dataset構築の統合テスト(列の網羅性・決定性・欠損Ticker除外)
- Hash関数の決定性・行順不変性
- **Leakageテスト**(AST検査・4種Future Shock Test・V1非依存性検査)

プロジェクト全体のtest suite: 927 passed / 2 deselected(既存875件 +
新規52件、regressionなし)。ruff/mypy(V3ソースコード対象、`tests/`は
本プロジェクトの既存慣行どおりmypyスコープ外)ともにクリア。

## 10. V1/V2 Non-Modification確認

`git status`で確認: 本Phaseで変更した既存追跡ファイルは`.gitignore`
(`data/v3/`の除外ルール追加)と`README.md`(V3節の追記のみ)の2つ
だけです。V1(`features/`・`signals/`・`scoring/`・`backtest/`・
`targets/`・`forward_test/`・`ensemble/`・`pipeline/`)・V2(`v2/`
配下全て)はバイト単位で無変更です。新規追加はすべて`v3/`・
`tests/test_v3_*.py`・`scripts/build_v3_dataset.py`・本報告書のみ
です。

## 11. 既知の制約・次Phaseへの申し送り

- **lightgbm/scipy/scikit-learnは現在未インストール**です。V3-2
  (Baseline ML)開始前に依存関係の追加が必要です(本Phaseでは
  インストール・使用していません)。
- Sector Feature(業種)は`data/reference/jpx_master_current.xls`の
  現在時点スナップショットに依存しており、V2-3で既に文書化した
  survivorship-bias相当の注意点が同様に当てはまります。Conditional
  扱いとし、デフォルトのDataset構築には含めていません。
- Full Universe(2,880銘柄)でのDataset生成・Walk Forward構造の
  実データ確認・ハイパーパラメータチューニング方針の確定は、いずれも
  次Phase(V3-2以降)の課題です。
- Target Registryの4variant中どれを最終採用するかは未決定のまま
  保持しています。

## 12. 停止

指示書section 38の明示的な指示どおり、本Phaseはここで停止します。
Full Universe OOS・Hyperparameter tuning・Feature selection・
Risk-adjusted Rankingの最終最適化・Streamlit UI・Paper Trading・
自動発注のいずれにも進んでいません。次のステップ(V3-2: Baseline
ML)は、本報告書のレビュー後、明示的な指示を受けてから開始します。

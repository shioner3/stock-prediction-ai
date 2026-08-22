# Phase 11A 完了報告: GitHub Actions完全自動Forward Test化

対象リポジトリ: [shioner3/stock-prediction-ai](https://github.com/shioner3/stock-prediction-ai)

Signal/Score/Backtest/Decision Frameworkのロジックは一切変更していません。本Phaseで行ったのは、(1) 既存のForward Test CLIをGitHub Actionsから起動する仕組みの構築、(2) その過程で発見した2件の実装バグの修正、の2点のみです。

以下、仕様書section 34の項目A〜Hに従って報告します。

---

## A. GitHub Actions

- Workflow名: `Forward Test daily run`
- ファイル: [`.github/workflows/forward_test.yml`](https://github.com/shioner3/stock-prediction-ai/blob/main/.github/workflows/forward_test.yml)
- schedule: `0 12 * * 1-5`(UTC、月〜金 12:00 = JST 21:00)
- workflow_dispatch: 対応済み(手動実行可能、実際に3回実行して検証)
- permissions: `contents: write`のみ(必要最小限、Secretsは不要 — yfinanceは認証不要)
- concurrency制御: 同時実行防止(`group: forward-test`)を設定

---

## B. Forward Test

- 実行コマンド: `python scripts/run_forward_test_day.py`(既存CLI、無変更)
- Data Fetch: 既存`pipeline.universe_ingest.run_universe_ingest()`をそのまま使用。実測でCandidate 3,713銘柄→Final Universe 2,781銘柄、fetch success 3,621/partial 92/failed 0、所要時間約1,001秒
- SAFE_ABORT: `pipeline/run_forward_test.py`の既存6種類の理由コードをそのまま使用、無変更
- Strategy Hash: `forward_test/manifest.py::verify_strategy_hashes_unchanged()`をそのまま使用(後述のバグ修正あり)
- Signal detection / Score: 既存`build_features`/`build_signals`/`build_scores`をそのまま使用。実測でFeature 2,781銘柄成功、Signal 674,960件trigger、Score 674,960件算出
- Portfolio / Position / Daily Performance: 既存`forward_test/portfolio.py`・`forward_test/performance_log.py`をそのまま使用、無変更

---

## C. Persistence

- 保存先: 同一リポジトリの`data/forward_test/`配下(`manifest.json`・`signals_log/`・`portfolio/`・`performance_log/`・`daily/`・`trades/`・`reports/`)
- 保存形式: 既存Phase 10/11の設計をそのまま維持(JSON/JSONL、追記専用)
- commit/push方式: workflow内でGitHub Actions Bot名義(`forward-test-bot <actions@users.noreply.github.com>`)でcommit・push。差分が無い日(SAFE_ABORT等)は自動的にcommitをスキップ
- 翌日からのstate復元方法: 毎回`actions/checkout@v4`でリポジトリを新規checkoutするため、前回commit済みのstateがそのまま読み込まれる。日次OHLCV/Feature/Signal/Scoreの再取得キャッシュ(`data/forward_test/{raw,processed,features,signals,scores}/`)は意図的に`.gitignore`で除外し、毎回フル再取得する設計(Phase 10/11の既存仕様通り)

---

## D. Safety

| 状況 | 挙動 | 検証状況 |
|---|---|---|
| stale data | `SAFE_ABORT[STALE_THRESHOLD_EXCEEDED]`、状態変更ゼロ | **実環境で確認済み**(後述) |
| hash mismatch | `StrategyHashMismatchError`でjob失敗 | **実環境で確認済み**(後述、意図せず本物のバグとして発生) |
| missing data | 同上のSAFE_ABORT機構でカバー | 設計上同一パス、ローカルテストで確認済み |
| duplicate execution | Signal Log/Portfolio/Performance Logいずれも追記専用でキー重複を検出しskip | Phase 10/11のローカルpytestで確認済み。GitHub Actions実環境での「同一日2回workflow_dispatch」は未実施(下記「今後可能なこと」参照) |
| unexpected failure | 非ゼロ終了コード(0でも2でもない)でjob失敗、`$GITHUB_STEP_SUMMARY`に明記 | **実環境で確認済み**(`git add`のパスバグがこの経路で正しく検出・可視化された) |

---

## E. 実環境テスト

実際にGitHub Actions上でworkflow_dispatchを3回実行し、2件の実装バグを発見・修正した:

### Run 1(失敗): Strategy Hash不一致
```
STRATEGY_HASH_MISMATCH: CONFIG/CODE CHANGED since Forward Test T0 -
mismatched fields: ['features_hash', 'signals_hash', 'scoring_hash',
'backtest_engine_hash', 'market_regime_hash', 'config_hash']
```
6項目全てが同時に不一致 — 原因は`common/hashing.py::hash_files()`が`str(path)`をハッシュ入力に使用しており、Windows(`\`区切り)とLinux(`/`区切り)でOS依存の差異が生じるバグ(詳細はREADME「Phase 11A: GitHub Actions化 + Strategy Hashバグ修正」節)。`Path.as_posix()`ベースに修正し、Strategy Hash対象38ファイルがバイト単位で無変更であることを独立検証した上でmanifestを再生成(Strategy Version 1のまま、T0不変)。

### Run 2(失敗): git addパスエラー
```
fatal: pathspec 'data/forward_test/reports/' did not match any files
Error: Process completed with exit code 128.
```
Strategy Hashチェックは通過(バグ修正が実環境で機能したことの実証)。原因は`data/forward_test/reports/`が一度もファイルを書き込まれたことがなく、Gitに追跡されていないため新規checkoutに存在しないこと。`mkdir -p`を追加して修正。

### Run 3(成功): SAFE_ABORT
```
2026-08-22 07:33:28 universe ingest complete: candidates=3713 attempted=3713
  success=3621 partial=92 failed=0 processed=2781 duration=1001.0s
2026-08-22 07:34:41 feature build complete: tickers=2781 success=2781 duration=73.0s
2026-08-22 07:35:09 signal build complete: tickers=2781 success=2781 triggered=674960 duration=28.1s
2026-08-22 07:36:03 score build complete: tickers=2781 success=2781 scored=674960 duration=53.6s
SAFE_ABORT[STALE_THRESHOLD_EXCEEDED]: 2781/2781 tickers (100%) are stale, threshold is 50%
```
Universe取得からScore算出まで全パイプラインが正常完了(所要時間合計約19分)した上で、当日分の市場データがまだ提供されていないことを正しく検知しSAFE_ABORTで安全停止。Workflow自体のjob conclusionは`success`(SAFE_ABORTは失敗として扱わない設計通り)。`git add`→差分なし→commitスキップも正しく動作し、リポジトリへの誤ったcommitは一切発生していない。

- workflow_dispatch: 3回とも正常に手動起動できることを確認
- 再実行: Run1→Run2→Run3といずれも独立して正常に起動、前回の失敗が後続実行に影響しないことを確認
- state persistence: 3回とも同一の`data/forward_test/manifest.json`(修正後)を正しく読み込んで実行していることを確認(Strategy Hash検証がRun2/3で一貫して通過)

---

## F. Strategy Integrity

- Signal: 無変更(12 Signal全て、条件・閾値とも触れていない)
- Score: 無変更(weight・threshold・bucket定義とも無変更)
- Backtest: 無変更(Entry/Exit/HOLD_DAYS/Cost tier全て無変更)
- Market Regime: 無変更
- Strategy Hash: **計算方式のバグを修正**(項目E参照)。対象ファイルの内容は独立したSHA256フィンガープリント比較および`git status`で1バイトも変化していないことを二重に確認済み。Strategy Version 1のまま、T0(2026-08-20)も不変

修正前後のHash値:

| フィールド | 修正前 | 修正後 |
|---|---|---|
| features_hash | `20c2748a...` | `78b72c8f...` |
| signals_hash | `20a79921...` | `1e113fe1...` |
| scoring_hash | `f3c75103...` | `6bda22b1...` |
| backtest_engine_hash | `8c7a8fb0...` | `ff8c51fa...` |
| market_regime_hash | `23d037fe...` | `8b4fffd1...` |
| config_hash | `a9b34ccb...` | `5e98a8c6...` |

---

## G. テスト

- `pytest`: **679 passed, 2 deselected**(Phase 12終了時点676件から+3件、hash platform-independence回帰テスト)
- `ruff check .`: All checks passed
- `mypy`(変更ファイル): Success、エラーなし

---

## H. Git

- リポジトリ: `shioner3/stock-prediction-ai`(既存プロジェクトはユーザー側で事前にクリア済みだったため、新規追加として構成)
- ブランチ・PR:
  - `add-swing-trading-scanner` → PR #1(マージ済み): Swing Trading Scanner本体(Phase 1-12、218ファイル)
  - `add-forward-test-workflow` → PR #2(マージ済み): `.github/workflows/forward_test.yml`新規追加
  - `fix-strategy-hash-path-separator-bug` → PR #3(マージ済み): Strategy Hashバグ修正
  - `fix-forward-test-reports-dir-git-add` → PR #4(マージ済み): git addパスバグ修正
- 現在の`main`最新commit: `c6cdc75`

---

## 既知の課題・今後可能なこと

1. **同一日2回workflow_dispatchを実行する冪等性テスト(仕様section 19 Test B)は、GitHub Actions実環境ではまだ未実施です。** Phase 10/11のローカルpytestでは確認済みですが、実環境での直接証明は次回の手動実行時に可能です。
2. **Open Position継続(仕様Test F)も同様に、実際に複数営業日にまたがるGitHub Actions実行ではまだ観測していません。** これはT0(2026-08-20)以降のSignal Log 4件(5858/598A/6367/7061)がEntry可能になるタイミングで自然に検証されます。
3. 今回の3回のworkflow_dispatch実行はいずれもStrategy Hashバグ・パスバグの発見、またはSAFE_ABORT(市場データ未到達)に終わり、**「実際にPaper Portfolio状態が更新されてcommit・pushされる成功例」はまだ観測できていません。** 日次スケジュール実行(平日21:00 JST)が進むにつれて自然に観測される見込みです。
4. GitHub Actions無料枠の消費量: 1回の実行で約20分(SAFE_ABORTの場合)〜さらに長め(Portfolio更新まで進む場合)。月20営業日換算で400分程度、GitHub Free/Proプランの無料枠(2,000分/月)内に収まる見込みですが、継続的な監視を推奨します。

---

## 重要な注意

- 本Phaseで発見・修正した2件のバグ(`common/hashing.py`のOS依存パス区切り、workflow内`git add`のパス欠損)はいずれも**インフラ・自動化の実装バグ**であり、Signal/Score/Backtest/Decision Frameworkの内容には一切影響していません(項目Fで独立検証済み)。
- Forward Testはまだ実運用実績が極めて浅く(実質1営業日+今回のSAFE_ABORT)、**戦略の有効性については一切結論を出しません。**
- Claude Codeの契約に依存せず、GitHub ActionsのみでForward Testが継続できる状態になったことを確認しましたが、上記「今known課題」の通り、まだ完全な多日サイクルでの動作は観測できていません。次回以降の日次自動実行を見守ることを推奨します。

# Phase 11 完了報告

Phase 11は明示的に2つの独立したTrackで構成される。**TrackA(Forward Test自動日次実行)とTrackB(残り11 Signal検証)は互いの結果を一切参照しない**。本報告書もこの分離を維持し、AとBを混在させた結論は一切述べない。

以下、指定された11項目のフォーマットに厳密に従う。

---

## 1. 変更ファイル一覧

### Track A(Forward Test)

**新規作成**
- `forward_test/performance_log.py` — Daily Performance Log(append-only JSONL)
- `tests/test_forward_test_performance_log.py`(11 tests)

**変更(Frozen対象外 — `forward_test/`, `pipeline/run_forward_test.py`, `scripts/`はStrategy Hashの対象に含まれない)**
- `forward_test/portfolio.py` — `OpenPosition`データクラス、`compute_open_positions()`、`Position.quantity`フィールド追加
- `pipeline/run_forward_test.py` — SAFE_ABORT実装、Open Position計算組み込み、Performance Log書き込み組み込み(全面改訂)
- `scripts/run_forward_test_day.py` — `SafeAbortError`ハンドリング、Open Position/Performance Logのサマリ出力追加
- `tests/test_forward_test_portfolio.py`(+7 tests: `compute_open_positions`系)
- `tests/test_forward_test_integrity.py`(既存ヘルパーに`quantity`フィールド追加、破壊的変更の修正)
- `tests/test_pipeline_run_forward_test.py`(T0混入回帰テスト1件 + SAFE_ABORTテスト4件を追加、既存呼び出し全箇所を新シグネチャに追従)

### Track B(Research)

**新規作成**
- `pipeline/run_phase11_research.py` — 残り11 Signal検証オーケストレーター
- `scripts/run_phase11_research.py` — CLIエントリポイント
- `tests/test_pipeline_run_phase11_research.py`(11 tests)

**変更**
- `pipeline/run_phase8_analysis.py` — `combined_report`引数を追加(オプション、デフォルト`None`で既存動作は完全不変)。Phase 11 Researchが`run_walk_forward()`の結果を11 Signal分使い回すための注入ポイント。既存Phase 8テスト9件は無変更で全通過を確認済み。
- `tests/test_phase9_no_lookahead.py` — `pipeline.run_phase11_research`を依存方向チェックの対象に追加

### 両Track共通の基盤バグ修正

- `backtest/bootstrap.py` — Bootstrap再標本化のメモリ確保をチャンク分割方式に変更(詳細は項目10「既知の課題」参照)
- `tests/test_backtest_bootstrap.py`(+3 tests: チャンク分割時の結果一致性、メモリ安全性)

**変更していないもの**: `features/`, `signals/`, `scoring/`, `backtest/engine.py`, `backtest/market_regime.py`, `config/settings.yaml`, `config/universe_filters.yaml` — Strategy Hashの対象は一切変更していない。

---

## 2. Forward Test実行方式

- **手動/スケジュールCLI実行**: `python scripts/run_forward_test_day.py [--run-date YYYY-MM-DD]`
- **GitHub Actions定期実行は実装していない** — 本プロジェクトのディレクトリはgitリポジトリとして初期化されておらず(リモートリポジトリなし)、GitHub Actionsのランナーを紐づける前提が満たされないため技術的に不可能と判断した。仕様書自身が「困難な場合は安全なCLI再実行手段」を代替として明示的に許容しているため、この代替方式を採用する。
- **安全な再実行手段(フォールバック)**: `run_forward_test_day()`はSignal Log・Paper Portfolio・Performance Logいずれも「日付/キー単位で既存なら書き込みをスキップ」という冪等設計になっており、同一日を何度再実行しても重複や上書きは発生しない(実データで検証済み — 項目4参照)。Windows タスクスケジューラや外部cronから同一コマンドを毎営業日実行するだけで安全に運用できる。

---

## 3. Strategy Hashの維持状況

- T0=2026-08-20時点のStrategy Hash(features/signals/scoring/backtest.engine/market_regime/config)は**Phase 11を通じて一度も変更していない**。
- 2026-08-20・2026-08-21の実データ実行いずれも `strategy_hash_unchanged: True` を確認。
- Track B(Research)の実行がStrategy Hash / Forward Test manifestに一切影響しないことを、実データではなく合成データによる専用の回帰テストで直接証明済み(`test_run_phase11_research_does_not_alter_forward_test_strategy_hash`)。

---

## 4. Forward Test実行結果(実データ)

**2026-08-20(T0)再実行(冪等性検証)**
- `strategy_hash_unchanged: True`
- `universe_candidate_count: 3713` → `final_universe_count: 2780`
- `fetch success/partial/failed: 3622/91/0`
- `new_signal_log_entries: 0`(既存4件のSignal — 5858, 598A, 6367, 7061 — と完全一致、重複なし)
- `new_closed_positions: 0`, `open_positions: 0`(T0翌日データがまだ無いため — 正しい挙動)
- `portfolio_equity: 10,000,000`(変化なし)
- `data_integrity_issues: 4銘柄`(いずれも軽微なstale、全体の50%閾値には遠く及ばず非ブロッキング)
- `trading_integrity.is_clean: True`
- `performance_log_written: True`(Performance Logへの初回書き込みを確認)

**2026-08-21(新規営業日)実行**
- **SAFE_ABORT[STALE_THRESHOLD_EXCEEDED]が正しく発火**: `2782/2782銘柄(100%)がstale`、閾値50%を大幅超過のため安全停止。
- この結果、Signal Log・Paper Portfolio・Performance Logのいずれにも一切の書き込みが発生していないことを確認(`daily/2026-08-21.json`も生成されず)。
- これはバグではなく、Forward Testのデータソースが「今日」時点でまだ当日データを提供できていない状況を正しく検知し、不完全なデータでの実行を防いだ**安全機構の実地動作確認**である。翌営業日以降、データソースが追いつき次第、再実行すれば正常に処理される設計になっている(冪等設計のため、失敗した日を再実行するだけでよい)。

---

## 5. Research対象の11 Signal

`long_oversold_rebound`(Strategy Version 1として凍結・Forward Test対象)を除く全Signal:

| Direction | Signal |
|---|---|
| LONG | long_breakout |
| LONG | long_ma_rebound |
| LONG | long_momentum_continuation |
| LONG | long_pullback |
| LONG | long_volume_breakout |
| SHORT | short_breakdown |
| SHORT | short_ma_rejection |
| SHORT | short_momentum_continuation |
| SHORT | short_overbought_reversal |
| SHORT | short_pullback |
| SHORT | short_volume_breakdown |

データセット: `data/phase7/`(Prime+Standard+Growth 2,880銘柄、2022-01-04〜2026-08-20)。新規データ取得は行わず、Phase 7で既に取得済みのデータを再利用した。

---

## 6. 各SignalのDecision

`backtest/decision.py::classify()`(Phase 6で固定された基準、一切変更なし)による機械的判定。**全11 Signal中、11 REJECT / 0 ACCEPT_CANDIDATE / 0 INSUFFICIENT_EVIDENCE**。

| Signal | Decision | n_oos | PF(base) | PF(high cost) | Expectancy CI(base) |
|---|---|---:|---:|---:|---|
| long_breakout | REJECT | 51,097 | 0.832 | 0.668 | [-0.00476, -0.00347] |
| long_ma_rebound | REJECT | 49,248 | 0.900 | 0.687 | [-0.00245, -0.00137] |
| long_momentum_continuation | REJECT | 110,757 | 0.868 | 0.687 | [-0.00341, -0.00259] |
| long_pullback | REJECT | 58,519 | 0.966 | 0.771 | [-0.00133, -0.00022] |
| long_volume_breakout | REJECT | 31,139 | 0.907 | 0.765 | [-0.00381, -0.00177] |
| short_breakdown | REJECT | 38,382 | 0.637 | 0.498 | [-0.00973, -0.00844] |
| short_ma_rejection | REJECT | 46,242 | 0.804 | 0.593 | [-0.00405, -0.00307] |
| short_momentum_continuation | REJECT | 97,080 | 0.705 | 0.550 | [-0.00735, -0.00655] |
| short_overbought_reversal | REJECT | 17,582 | 0.761 | 0.595 | [-0.00652, -0.00454] |
| short_pullback | REJECT | 55,193 | 0.653 | 0.513 | [-0.00949, -0.00840] |
| short_volume_breakdown | REJECT | 26,306 | 0.708 | 0.584 | [-0.00998, -0.00800] |

全SignalでExpectancy信頼区間の上限がゼロ未満、またはゼロ近傍で優位性を示せていない。長期側5 Signal・ショート側6 Signal全てが同じ結論に至っており、特定方向への偏りは見られない。

---

## 7. 統計的検証結果

**Combined OOS Permutation Test(全12 Signal同時、min_oos_start=None、Walk Forward 14 window、2022-01-04〜2026-08-20)**

| Signal | raw p-value | FDR-adjusted q-value(全12 Signal中) |
|---|---:|---:|
| long_breakout | 1.0000 | 1.0000 |
| long_ma_rebound | 1.0000 | 1.0000 |
| long_momentum_continuation | 1.0000 | 1.0000 |
| long_pullback | 0.5674 | 1.0000 |
| long_volume_breakout | 1.0000 | 1.0000 |
| short_breakdown | 0.1006 | 0.4024 |
| short_ma_rejection | 1.0000 | 1.0000 |
| short_momentum_continuation | 0.4451 | 1.0000 |
| short_overbought_reversal | 0.6647 | 1.0000 |
| short_pullback | 0.2345 | 0.7035 |
| short_volume_breakdown | 0.0541 | 0.3246 |

最小のraw p-valueは`short_volume_breakdown`の0.0541で、有意水準0.05をわずかに上回っており、FDR補正後(q=0.3246)では明確に非有意。多重検定の問題を差し引いても「見せかけの優位性」に該当するSignalは存在しない。

**その他の統計的検証(全Signal・全項目実施)**
- Bootstrap信頼区間(mean_return / profit_factor / expectancy、n_resamples=10,000)
- BEAR regime制限Permutation Test(Phase 8ロジック再利用)
- Day Cluster Bootstrap / Block Bootstrap(BEAR・Combined各々)
- Leave-One-Year-Out / Leave-One-BEAR-Episode-Out(Bootstrap CI付き)
- Timing Placebo sweep(オフセット `-15,-10,-5,-3,-1,0,+1,+3,+5,+10` — Phase 9のデフォルトに 0/+1/+3 を追加した仕様書指定の10点、実行時に確認済み)
- Cost sensitivity(zero/low/base/highの4 tier)
- Forward Holding Period profile(1/3/5/10/20日)
- Winsorize・8月2024イベント除外・年別除外の6シナリオ(A/B/C/E/F — Dはepisode LOPOで代替)

---

## 8. Event/Regime/Placebo結果

**Day-level Event concentration(Combined OOS、`backtest/event_concentration.py`)**

Gini係数は`long_pullback`(8.6)・`short_breakdown`(3.6)など比較的低い(=分散した)ものから、`long_volume_breakout`(86.4)のように極端に集中しているものまで幅広い。ただし集中度の高低はDecisionを左右していない — 分散していても集中していてもいずれもREJECTという結論は変わらない。

**BEAR regime Placebo negative control(Signal自身の日付をtrading-day単位で15日分shiftした対照実験)**

多くのSignalで「本物のBEAR成績」より「Placeboの方が高いPF」を示すケースが目立つ(例: `long_breakout` real=0.968 vs placebo=3.666、`long_momentum_continuation` real=1.325 vs placebo=4.420)。これは「BEAR局面で単にLONGを持つ/SHORTを持つこと自体」がSignal固有の優位性ではなく市場全体の動きに由来する可能性を示唆しており、REJECT判定と整合的である。

**Cross-scenario(A/B/C/E/F)**

8月2024イベント除外(B)・2024年除外(C)・Winsorize(E/F)いずれの操作でもPFが1を明確に上回るSignalは現れなかった。特定の期間・イベントに依存して結論が変わっている様子はない。

---

## 9. テスト結果

- `pytest`: **622 passed, 2 deselected**(Phase 10終了時点587件から+35件)
- `ruff check .`: All checks passed
- `mypy`(Phase 11で変更した各ファイル個別): 全てSuccess、エラーなし
- Phase 11で追加した主なテスト:
  - T0混入防止の明示的回帰テスト(`test_run_forward_test_day_excludes_pre_t0_signals_and_trades`)
  - SAFE_ABORT 4条件(UNIVERSE_DATA_INCOMPLETE / MARKET_DATA_UNAVAILABLE / STALE_THRESHOLD_EXCEEDED / PORTFOLIO_STATE_CORRUPTION)
  - Open Position計算(`compute_open_positions`)の単体テスト7件
  - Daily Performance Logの冪等性・累積計算テスト11件
  - Track B: `run_walk_forward()`が11 Signal分の呼び出しでも**厳密に1回しか呼ばれない**ことを直接検証するテスト、Strategy Hash不変性テスト、Research offsetsが仕様通り10点で実行されることの検証
  - Bootstrap チャンク分割の結果ビット完全一致テスト・大規模データでのメモリ安全性テスト

---

## 10. 既知の課題

1. **GitHub Actions未実装**(項目2参照) — gitリポジトリ化されていないため技術的に不可能。将来gitリポジトリ化・リモート接続する場合は`.github/workflows/`の追加で対応可能。
2. **`backtest/bootstrap.py`のメモリスケーラビリティ問題を本Phaseで発見・修正** — `long_momentum_continuation`など高頻度発火Signalの forward-return母集団(47万行規模)に対し、既存のBootstrap実装が `(n_resamples, n)` の巨大配列を一括確保しようとして35GBのメモリ確保エラーで停止した。これはPhase 6-9では`long_oversold_rebound`(まれにしか発火しない)でしか使われていなかったため潜在化していたバグ。チャンク分割方式に修正し、既存の全テスト結果とビット完全一致することを専用テストで証明した上で再実行し、正常完了を確認した。**Signal/Score/Decisionロジックには一切手を加えていない** — 純粋な実装上のメモリ管理バグの修正である。
3. **`mypy .`をリポジトリ全体に対して実行すると、Phase 11で触れていない既存テストファイル9件に計38件の型エラーが検出される**(`test_backtest_metrics.py`, `test_universe.py`, `test_providers.py`ほか)。いずれもPhase 11以前から存在する、テストコード内の`float | None`型の扱いに関する軽微なもので、Phase 11で変更したファイルには一切含まれない。スコープ外のため本Phaseでは修正していない。
4. Forward Testは実データでまだ2営業日分(T0 + stale abort 1件)しか実行できておらず、有意な運用実績にはまったく達していない(項目11で後述)。

---

## 11. 次Phaseで可能なこと

- Forward Testの日次実行を継続し、実際の約定・Paper Portfolio推移・Performance Logを蓄積する(仕様書が定める最低6-12ヶ月の観測期間に向けて)。
- 今回REJECTと判定された11 Signalについて、Signal条件・Score・閾値を一切変更しない前提であれば、これ以上の追加検証は不要(十分な統計的根拠を持ってREJECTに至っている)。
- 仕様書の停止条件により、Streamlit UI・実運用自動発注・自動売買・Strategy Version 2・新規Signal発明・Score調整のいずれにも、本Phaseの範囲を超えて着手しない。

---

## 重要な注意(Track A/Bの分離、および結論の限界)

- 本報告書のTrack A(項目2-4)とTrack B(項目5-8)は完全に独立して実施されており、**一方の結果が他方に影響を与えたことは一切ない**(項目3に技術的な非干渉性の検証結果を記載)。
- Track Aの実データ実行はまだ2営業日分に過ぎず、**戦略の有効性について一切の結論を出さない**。SAFE_ABORTが正しく発火したことは実装の正しさの確認であり、`long_oversold_rebound`自体の収益性を示すものではない。
- Track Bの11 REJECTという結果は「今回入手可能なデータの範囲でこれらのSignalに統計的優位性を確認できなかった」という検証結果であり、将来のいかなる市場環境でも通用しないことを断定するものではない。ACCEPT_CANDIDATE同様、REJECTも人間が最終判断する際の一つの参考情報である。

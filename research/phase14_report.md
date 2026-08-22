# Phase 14 完了報告(実行前STOP): long_oversold_rebound Conditional Hypothesis Confirmatory OOS Validation

既存12 Signal・Score・Backtest・Decision Framework・Strategy Version 1は一切変更していません。本Phaseは**Full Universe本実行を開始せず**、指示書section 20の実行前CHECKPOINTのみを実施し、section 3の独立OOS確保条件が満たせないことを確認した時点でSTOPしました。

---

## 1. 目的(指示書より)

Phase 13で探索的に発見された「long_oversold_reboundはBEAR regimeかつ市場が大幅下落している局面でのみ優位性が強まる」という仮説(H1〜H5)を、Phase 13の分析結果を一切利用した追加最適化を行わず、**Phase 13の分析対象期間と重複しない完全独立OOS**で検証すること。

## 2. 結論

**INSUFFICIENT_EVIDENCE — 実行不能。現時点で利用可能なデータでは、Phase 13の分析期間と重複しない十分な長さの独立OOS期間を確保できません。** 指示書section 3自身が定める停止条件(「完全独立OOSを十分な長さ確保できない場合は、実行せずSTOPして報告する」)に該当したため、Full Universe本実行(数時間規模の計算)は一切開始していません。

---

## 3. 実行前CHECKPOINT結果(指示書section 20の12項目)

| # | 項目 | 結果 |
|---|---|---|
| 1 | Phase 13実装内容の確認 | 完了。`pipeline/run_phase13_conditional_analysis.py`が対象。 |
| 2 | Phase 14スクリプトのレビュー | 完了。後述のとおり、本Phaseの目的(独立OOS確認)には使用できないと判断。 |
| 3 | config確認 | `config/settings.yaml`/`config/universe_filters.yaml`、無変更を確認。 |
| 4 | dataset期間確認 | 下記4節参照。 |
| 5 | OOS期間確認 | **NG — 独立OOSが確保できない**(下記5節参照)。 |
| 6 | hypothesis固定値確認 | H1〜H5は指示書に明記された通り(BEAR、TOPIX20d<=-10%等)、Phase 13の結果を使った後付け変更なし。 |
| 7 | Strategy Hash確認 | **一致**。`verify_strategy_hashes_unchanged()`を実データ(`data/forward_test/manifest.json`)に対して実行、mismatchなし。Strategy Version = v1、T0 = 2026-08-20。 |
| 8 | Phase 10/11 Forward TestのHash確認 | 同上(7と同じ検証で兼ねる)。Forward Testは完全凍結のまま、本Phaseからの変更なし。 |
| 9 | Phase 13とのデータ重複確認 | **重複(というより完全な包含関係)を確認**(下記5節参照)。 |
| 10 | pytest実行 | 711 passed / 2 deselected。 |
| 11 | ruff実行 | All checks passed。 |
| 12 | mypy実行 | pipeline/backtest/config/scripts配下57ファイルでエラーなし。 |

## 4. dataset期間(実データを直接確認)

- `config/settings.yaml`の`start_date`: 2018-01-01(Feature warmup用の生データ取得開始日、実際のWalk Forward分析開始は2022年以降)。
- Phase 7 WFO windowsの最終ウィンドウ: `oos_start: 2026-07-04`, `oos_end: 2026-08-20`(`oos_truncated: True`)— これが現在保持している全履歴データの終端。
- Forward Test T0: **2026-08-20**(Phase 10で凍結・記録済み、`data/forward_test/manifest.json`)。
- Full Universe銘柄数: 2,880銘柄(`data/phase7/_universe_fetch_manifest.json`の`included_in_universe`件数、Phase 6.5/7/12/13と同一)。

## 5. OOS独立性の判定(本Phase最大の論点)

Phase 13の保存済みレポート(`data/walk_forward/phase13_conditional_report.json`)を実際に読み込み、Phase 13が識別したBEAR episodeの一覧を確認しました:

| episode開始 | episode終了 | n(core trades) |
|---|---|---:|
| 2022-04-11 | 2022-04-12 | 11 |
| 2022-06-21 | 2022-06-23 | 65 |
| 2022-07-01 | 2022-07-01 | 0 |
| 2024-08-02 | 2024-08-15 | 652 |
| 2024-08-19〜2024-10-16(複数episode) | | 85 |
| **2025-04-03** | **2025-04-17** | 398 |
| 2025-04-21 | 2025-04-24 | 1 |
| **2026-03-30** | **2026-03-31** | 14 |

Phase 13が識別した最後のBEAR episodeは**2026-03-31**であり、これはPhase 7の最終WFO windowの`oos_end`である2026-08-20に極めて近い時点まで、Phase 13が**既に全期間を対象に分析済み**であることを意味します。つまり、Phase 13は「2022年初頭からT0(2026-08-20)まで」というこのプロジェクトで取得済みの全履歴データを対象に分析しており、その外側に「Phase 13が見ていない、かつ十分な長さを持つ」過去データは存在しません。

T0以降(2026-08-20〜)はForward Test(Phase 10/11A)による日次収集の対象ですが、実データを確認したところ:

- `data/forward_test/daily/`に記録済みの日数: **1日のみ**(2026-08-20 = T0当日)
- `data/forward_test/performance_log/`のT0当日エントリ: `signal_count: 0`, `closed_positions: 0`, `realized_pnl: 0`(初日でエントリー済みポジションが未決済のため、当然の結果)
- 本日時点(2026-08-23)で、T0以降の追加日次実行が状態としてコミットされた形跡なし

したがって、Phase 13の分析期間と重複しない独立OOSとして現在使えるデータは実質ゼロ日です。指示書H1〜H5(特にH5「2025年4月の急落局面で観測された傾向が、独立した別期間でも再現する」)を検証するには、Phase 13がまだ見ていない新しい急落局面がForward Test期間中に発生し、統計的に意味のある件数のSignal発生・Trade決済が蓄積される必要があります。

## 6. 「実行前レビュー」で判明したもう一つの論点

本Phase着手前、私は本指示書とは異なる設計のPhase 14実装(`pipeline/run_phase14_validation.py`ほか、ブランチ`add-phase14-validation`にpush済み)を先に作成していました。これはPhase 13と**同一のCombined dataset**(2022年〜T0)に対して、より厳格な統計的頑健性チェック(Ticker Cluster Bootstrapの追加、Timing Placebo、Leave-One-Episode-Out/Leave-One-Year-Out、Event除外、Permutation+FDR等)を行うものであり、それ自体は正しく動作します(合成データによるテスト711件全てPASS、実装上のバグも1件発見・修正済み)。

しかし、これは本指示書が求める「Phase 13の分析対象期間と重複しない完全独立OOS」ではありません。同一データに対する再分析(頑健性の再検証)と、時間的に独立した新規データによる確認的検証(confirmatory OOS validation)は、統計的に別物です。前者は有用な副産物として実装済み・テスト済みのまま保持しますが、**本指示書のPrimary Analysisの代替にはなりません**。この区別を明確にした上で報告するよう、着手前に確認を取りました。

## 7. 禁止事項の遵守

- Signal/Score/Backtest/Regime/Decision Frameworkのいずれも変更していません。
- OOS期間・hypothesisの閾値を、結果を見てから調整する行為は一切行っていません(そもそも本実行に至っていません)。
- Forward Test側の状態・条件への変更は一切ありません。
- Strategy Version 2作成・自動発注・実資金投入・Streamlit UI作成のいずれも行っていません。

## 8. 今後について

独立OOSが確保できるようになるまで、本Phaseの本実行(Primary Analysis〜Final Report)は保留します。目安として、このプロジェクトが既存のForward Test運用方針として掲げている「最低6ヶ月(理想は12ヶ月以上)」の観測期間([[project_phase6_5_status]]参照)が、統計的に意味のある独立OOSを確保する上でも妥当な目安になると考えられます。それまでの間、Forward Testは`.github/workflows/forward_test.yml`により無人で日次運用を継続します。

再開する際は、その時点のForward Test蓄積データを使って独立OOS期間を機械的に(Phase 13の結果を見ずに)再決定し、本報告の section 3〜5 と同じ手順で確認してから、あらためてFull Universe本実行の可否を判断してください。

## 9. Classification

**INSUFFICIENT_EVIDENCE**(実行不能 — 独立OOSデータ不足)。「PFが高い」等の強い数値による判定は一切行っていません(そもそも算出していません)。

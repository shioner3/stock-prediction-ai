# PHASE 7 FINAL REPORT: 完全独立OOS検証(2024-07-01〜2026-08-20)

実行日: 2026-08-20
config_hash: `a9b34ccb6e6e1e1a9d4d1eb554fe3fd8a4c09b2d9d186b2bd0ee3b0d430a2217`
(**Phase 6.5と完全一致** — Signal/Score/Backtest/WFO設定が一切変更されて
いないことの直接的な証拠)
data_hash: `fd0b57df83a1a2b07e72e3856040edd41bea7d446f8184b7579da772dca55727`

---

## 0. Executive Summary

| 項目 | 値 |
|---|---|
| 対象OOS期間 | 2024-07-01 〜 2026-08-20(完全独立、Phase 6.5と無重複) |
| WFO Window数(OOS開始が2024-07-01以降) | 9(index 5〜13) |
| 評価Signal数 | 12(LONG 6 + SHORT 6、変更なし) |
| ACCEPT_CANDIDATE | **1**(LONG:long_oversold_rebound) |
| REJECT | 11 |
| INSUFFICIENT_EVIDENCE | 0 |
| 総トレード数(12 Signal合算、base tier) | 377,713 |
| Unique銘柄数(Final Universe) | 2,880 |

**最重要所見**: `long_oversold_rebound`がPhase 6.5(INSUFFICIENT_EVIDENCE)
からPhase 7で**ACCEPT_CANDIDATE**に変わった(Case D)。ただし内訳を見ると、
この結果は**ほぼ全面的にBEAR regime(PF=42.8、n=1,150)に依存**しており、
BULL regimeではPF≈1.0(ほぼフラット)、NEUTRAL regimeではPF<1(0.75)である。
「全期間で安定して効くSignal」ではなく「相場急落局面での押し目買いが
非常に効く」という、より限定的だが実務的には重要な発見である。
**この時点でも実運用への自動採用は行わない**(仕様20項の通り、人間による
採用判断が必須)。

他11 Signalは全てCase A(PF<1が両フェーズで再現、優位性を支持しない)、
Case B×0、Case C×1(long_pullback、Phase 6.5ではPF>1だったがPhase 7では
再現せず)という結果だった。

---

## 1. Dataset Summary

| 項目 | 値 |
|---|---|
| JPX Master候補数 | 4,444(Phase 6.5と同一スナップショットを再利用) |
| Static Filter通過(Prime+Standard+Growth) | 3,713 |
| Fetch試行 | 3,713 |
| Fetch成功 / 部分成功 / 失敗 | 3,397 / 316 / **0**(Phase 6.5は134失敗 -
  取得期間が2026-08-20まで延びたことで、以前は上場前だった新規上場銘柄も
  データを持つようになったため) |
| Price/Liquidity Filter除外 | 833 |
| **Final Universe** | **2,880銘柄**(Phase 6.5は2,755銘柄) |
| データ取得期間(実データ) | 2022-01-04 〜 2026-08-20(TRAIN/VALIDATION
  構築に必要な過去データを含む。Phase 7自身のOOS報告対象は
  2024-07-01以降のWindowのみ) |
| Coverage比率 平均 / 中央値 | 89.5% / 93.7% |
| Survivorship Bias | **警告あり**(§9のPhase 6.5と同一方針。Current
  Universe方式、過去の上場廃止銘柄は再構成不可) |

---

## 2. Per-Signal Summary(OOS 2024-07〜2026-08、Base cost tier)

| Signal | Dir | Trades | Unique銘柄 | Win Rate | PF(base) | Expectancy | Bootstrap CI | Perm. p | FDR adj.p | Windows PF>1 | Decision |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| long_breakout | LONG | 31,612 | 2,867 | 42.3% | 0.821 | -0.00460 | [-0.00546,-0.00375] | 1.000 | 1.000 | 2/9 | REJECT |
| long_ma_rebound | LONG | 31,379 | 2,861 | 45.6% | 0.868 | -0.00269 | [-0.00342,-0.00196] | 1.000 | 1.000 | 4/9 | REJECT |
| long_momentum_continuation | LONG | 70,262 | 2,870 | 43.7% | 0.860 | -0.00334 | [-0.00387,-0.00280] | 1.000 | 1.000 | 3/9 | REJECT |
| **long_oversold_rebound** | LONG | 5,548 | 2,144 | 54.3% | **1.800** | **+0.01488** | [+0.01277,+0.01696] | **0.0000** | **0.0000** | 5/9 | **ACCEPT_CANDIDATE** |
| long_pullback | LONG | 37,090 | 2,824 | 46.4% | 0.936 | -0.00151 | [-0.00225,-0.00077] | 1.000 | 1.000 | 5/9 | REJECT |
| long_volume_breakout | LONG | 19,657 | 2,811 | 43.6% | 0.926 | -0.00234 | [-0.00367,-0.00099] | 1.000 | 1.000 | 3/9 | REJECT |
| short_breakdown | SHORT | 25,092 | 2,866 | 42.6% | 0.632 | -0.00992 | [-0.01077,-0.00906] | 0.0000 | 0.0000 | 1/9 | REJECT |
| short_ma_rejection | SHORT | 29,938 | 2,865 | 46.9% | 0.875 | -0.00224 | [-0.00284,-0.00163] | 1.000 | 1.000 | 2/9 | REJECT |
| short_momentum_continuation | SHORT | 62,214 | 2,873 | 44.2% | 0.742 | -0.00626 | [-0.00678,-0.00574] | 0.0000 | 0.0000 | 1/9 | REJECT |
| short_overbought_reversal | SHORT | 10,888 | 2,631 | 45.6% | 0.783 | -0.00512 | [-0.00643,-0.00383] | 0.9139 | 1.000 | 4/9 | REJECT |
| short_pullback | SHORT | 36,388 | 2,862 | 43.2% | 0.610 | -0.01085 | [-0.01156,-0.01014] | 0.0000 | 0.0000 | 2/9 | REJECT |
| short_volume_breakdown | SHORT | 17,645 | 2,853 | 45.5% | 0.670 | -0.01110 | [-0.01237,-0.00980] | 0.0000 | 0.0000 | 1/9 | REJECT |

4つのSHORT Signal(breakdown/momentum_continuation/pullback/volume_breakdown)
はpermutation p=0.0000で「統計的に有意」だが、これは**有意に悪い**方向
(観測平均が母集団よりも大きく損失側に乖離)であり、有意性=優位性ではない
ことに注意。Decision(REJECT)は正しくこれを反映している。

---

## 3. Cost Sensitivity(Profit Factor、4 tiers)

| Signal | Zero | Low | Base | High |
|---|---:|---:|---:|---:|
| long_breakout | 0.933 | 0.894 | 0.821 | 0.666 |
| long_ma_rebound | 1.016 | 0.964 | 0.868 | 0.670 |
| long_momentum_continuation | 0.985 | 0.941 | 0.860 | 0.688 |
| **long_oversold_rebound** | **2.035** | **1.953** | **1.800** | **1.470** |
| long_pullback | 1.067 | 1.022 | 0.936 | 0.755 |
| long_volume_breakout | 1.022 | 0.989 | 0.926 | 0.788 |
| short_breakdown | 0.726 | 0.693 | 0.632 | 0.502 |
| short_ma_rejection | 1.047 | 0.986 | 0.875 | 0.650 |
| short_momentum_continuation | 0.856 | 0.816 | 0.742 | 0.586 |
| short_overbought_reversal | 0.904 | 0.862 | 0.783 | 0.618 |
| short_pullback | 0.700 | 0.669 | 0.610 | 0.485 |
| short_volume_breakdown | 0.747 | 0.721 | 0.670 | 0.559 |

`long_oversold_rebound`は**High cost tier(80bps)でもPF=1.47**と、唯一
コスト耐性を明確に示した。Phase 6.5ではHigh costでPF=0.895(<1)だった
ことと対照的。

---

## 4. Regime Analysis(BULL / NEUTRAL / BEAR)

TOPIX Proxy 60日トレーリングリターンによる分類。**Phase 7の期間には実際に
BEAR判定日が存在した**(2024-08-02〜2026-03-31の間に散発的に58営業日、
Phase 6.5では0件だった制約が今回は解消)。

| Signal | BEAR PF (n) | BULL PF (n) | NEUTRAL PF (n) |
|---|---|---|---|
| long_breakout | 0.954 (1,738) | 1.002 (18,190) | 0.838 (11,684) |
| long_ma_rebound | 1.771 (2,277) | 1.128 (18,020) | 0.783 (11,082) |
| long_momentum_continuation | 1.254 (4,740) | 1.025 (40,421) | 0.890 (25,101) |
| **long_oversold_rebound** | **42.816 (1,150)** | 0.997 (2,515) | 0.747 (1,883) |
| long_pullback | 1.407 (3,009) | 1.148 (20,134) | 0.917 (13,947) |
| long_volume_breakout | 2.816 (2,272) | 0.944 (9,986) | 0.807 (7,399) |
| short_breakdown | 0.180 (5,029) | 0.845 (11,032) | 1.496 (9,031) |
| short_ma_rejection | 0.825 (4,374) | 0.871 (14,512) | 1.401 (11,052) |
| short_momentum_continuation | 0.272 (9,685) | 0.864 (29,062) | 1.358 (23,467) |
| short_overbought_reversal | 1.204 (284) | 0.819 (7,067) | 1.053 (3,537) |
| short_pullback | 0.183 (9,191) | 1.001 (16,510) | 1.145 (10,687) |
| short_volume_breakdown | 0.189 (4,096) | 1.004 (7,541) | 1.414 (6,008) |

`long_oversold_rebound`のBEAR regime PF=42.8は、他のどのSignal・regime
組み合わせよりも桁違いに高い。n=1,150と統計的に無視できる規模ではない
ものの、「新規上場ラッシュのような特殊要因ではないか」「BEAR判定58営業日
中の特定の急落イベントに取引が集中していないか」は、本レポートのCase D
判定基準(§10)には含まれないため、**人間によるさらなる深掘りが必要な
オープン論点**として明記する。

興味深い対照として、SHORT系4 Signal(breakdown/momentum_continuation/
pullback/volume_breakdown)はBEAR regimeでPF 0.18〜0.27と極端に低い
(NEUTRAL/BULLでは1前後)。「下落相場でSHORTが儲からない」という直感に
反する結果に見えるが、これはEntry=Open[t+1]・HOLD_DAYS=5の固定保有期間
Backtestの下で、急落後の急反発(BEAR regimeの一部はまさに反発局面)を
SHORTポジションが直撃している可能性がある。Backtest仕様自体は本Phaseで
変更していない。

---

## 5. Score Q1-Q5 Validation(抜粋、long_oversold_rebound、forward 5d、quantile buckets)

| Bucket | n | Avg Return |
|---|---:|---:|
| Q1 | 9,394 | 0.01319 |
| Q2 | 1 | 0.08255 |
| Q3 | 1 | 0.10677 |

Quantile bucketはOOS全体の総合Score分布から算出しているため、
long_oversold_rebound自身のトリガー行はほぼ全てQ1相当に収まり、Q4/Q5には
1行ずつしか入らない(Phase 6.5と同じ既知の制約、§本文脚注参照)。
`monotonic=True, corr=0.963`という値はQ2/Q3が単一観測値であることに
起因する統計的アーティファクトであり、意味のあるScore単調性の証拠とは
見なさない。

---

## 6. Concentration Analysis(Top1/5/10)

| Signal | Unique銘柄 | Top1 trade | Top5 | Top10 | Top1 return | Top10 |
|---|---:|---:|---:|---:|---:|---:|
| long_breakout | 2,867 | 0.07% | 0.35% | 0.68% | -4.4% | -26.4% |
| long_ma_rebound | 2,861 | 0.08% | 0.38% | 0.74% | +26.9% | +143.1% |
| long_momentum_continuation | 2,870 | 0.07% | 0.33% | 0.66% | -11.1% | -80.5% |
| **long_oversold_rebound** | 2,144 | 0.32% | 1.33% | 2.47% | +1.0% | +7.0% |
| long_pullback | 2,824 | 0.11% | 0.51% | 0.98% | +4.9% | +31.3% |
| long_volume_breakout | 2,811 | 0.12% | 0.57% | 1.10% | +15.6% | +114.6% |
| short_breakdown | 2,866 | 0.08% | 0.40% | 0.79% | -0.5% | -3.8% |
| short_ma_rejection | 2,865 | 0.08% | 0.40% | 0.77% | +4.9% | +34.2% |
| short_momentum_continuation | 2,873 | 0.08% | 0.41% | 0.78% | -0.9% | -6.0% |
| short_overbought_reversal | 2,631 | 0.17% | 0.71% | 1.32% | -2.7% | -21.3% |
| short_pullback | 2,862 | 0.11% | 0.54% | 1.03% | -0.6% | -3.6% |
| short_volume_breakdown | 2,853 | 0.12% | 0.54% | 1.04% | -0.8% | -5.4% |

`long_oversold_rebound`のTop10 trade shareは2.47%(2,144銘柄に分散)で、
極端な集中は見られない。「少数銘柄の異常値がBEAR regime PF=42.8を作って
いる」という懸念は、この指標からは支持されない(ただし§4で述べた
イベント集中の可能性は別軸の論点として残る)。

---

## 7. Phase 6.5 vs Phase 7 Comparison

| Signal | Case | PF(base) 6.5→7 | Δ | Expectancy 6.5→7 | Decision 6.5→7 |
|---|---|---|---:|---|---|
| long_breakout | A | 0.855→0.821 | -0.034 | -0.0033→-0.0046 | REJECT→REJECT |
| long_ma_rebound | A | 0.961→0.868 | -0.093 | -0.0007→-0.0027 | REJECT→REJECT |
| long_momentum_continuation | A | 0.883→0.860 | -0.024 | -0.0024→-0.0033 | REJECT→REJECT |
| **long_oversold_rebound** | **D** | 1.160→**1.800** | **+0.640** | +0.0028→**+0.0149** | INSUFFICIENT_EVIDENCE→**ACCEPT_CANDIDATE** |
| long_pullback | C | 1.019→0.936 | -0.083 | +0.0004→-0.0015 | REJECT→REJECT |
| long_volume_breakout | A | 0.871→0.926 | +0.056 | -0.0036→-0.0023 | REJECT→REJECT |
| short_breakdown | A | 0.649→0.632 | -0.017 | -0.0075→-0.0099 | REJECT→REJECT |
| short_ma_rejection | A | 0.680→0.875 | +0.194 | -0.0060→-0.0022 | REJECT→REJECT |
| short_momentum_continuation | A | 0.632→0.742 | +0.111 | -0.0083→-0.0063 | REJECT→REJECT |
| short_overbought_reversal | A | 0.723→0.783 | +0.060 | -0.0062→-0.0051 | REJECT→REJECT |
| short_pullback | A | 0.759→0.610 | -0.148 | -0.0052→-0.0108 | REJECT→REJECT |
| short_volume_breakdown | A | 0.812→0.670 | -0.142 | -0.0047→-0.0111 | REJECT→REJECT |

Case内訳: **A(両フェーズでPF<1) = 10**、**B(regime依存の可能性、新規発見) = 0**、
**C(Phase 6.5のみPF>1、再現せず) = 1**(long_pullback)、**D(両フェーズで
PF>1、候補として扱う) = 1**(long_oversold_rebound)。

12 Signal中10がCase A(両フェーズ一貫してREJECT水準)という結果は、
「Phase 6.5のREJECT判定の大部分は期間固有のノイズではなく、より安定した
無効性の証拠である」ことを示唆する。

---

## 8. Integrity

- `config_hash` = `a9b34ccb6e6e1e1a9d4d1eb554fe3fd8a4c09b2d9d186b2bd0ee3b0d430a2217`
  — **Phase 6.5と完全一致**。`config/settings.yaml`・
  `config/universe_filters.yaml`を一切変更していないことの直接証拠。
- `data_hash` = `fd0b57df83a1a2b07e72e3856040edd41bea7d446f8184b7579da772dca55727`
  (Phase 7専用データセット、2,880銘柄分のFeature Parquet群のsha256結合
  ハッシュ。Phase 6.5のdata_hashとは意図的に異なる - 別データセットである
  ため)
- データ分離: `data/phase7/{raw,processed,features,signals,scores}` に
  完全に独立して保存。`data/{raw,processed,features,signals,scores}`
  (Phase 6.5以前のデータ)は一切上書きしていない。
- 再現性: `pipeline/run_walk_forward.py`の`min_oos_start`パラメータの
  決定性(TRAIN/VALIDATION日付が不変であること、Windowフィルタリングの
  正確性)を`tests/test_pipeline_run_walk_forward.py`の新規テストで直接
  検証済み。

---

## 9. Bugs / Fixes(Phase 7で発見した事象)

### 9.1 実バグ: `pipeline/universe_ingest.py`がTOPIX Proxy市場指数を一度も
取得していなかった

- **発見経緯**: Phase 7専用の新規ディレクトリ`data/phase7/processed/`に
  対して`run_walk_forward()`を実行したところ、Market Regime計算部分で
  `FileNotFoundError`によりクラッシュ。
- **原因**: `pipeline/universe_ingest.py::run_universe_ingest()`は
  `pipeline/ingest.py::ingest_one_ticker()`を個別銘柄ごとに呼ぶだけで、
  `run_ingest()`が末尾で行っている市場指数(TOPIX Proxy, 1306.T)の取得・
  保存処理が最初から実装されていなかった。
- **Phase 6.5以前の結果への影響評価**: **影響なし**。`data/processed/`
  には元々Phase 1の`run_ingest()`実行時に取得された本物の
  `TOPIX.parquet`が存在しており、Phase 6.5はこれをそのまま再利用して
  いた(欠落に気づかないまま正しく動作していた)。Phase 6.5の
  Relative Strength・Market Regime関連の数値は全て正しいデータに基づく。
- **修正**: `run_universe_ingest()`に市場指数フェッチ処理を追加(既存の
  `run_ingest()`のロジックをそのまま再利用、キャッシュ済みTOPIXファイルが
  存在する場合はスキップする点も含めて同じキャッシュ方式を踏襲)。
  `tests/test_pipeline_universe_ingest.py`に検証テスト4件を追加。
- **修正前後の比較**: 修正前はクラッシュして結果自体が得られなかったため
  数値比較は不可能。修正後、Phase 7のFeature構築ログで
  `market_data_available=True`(修正前は`False`)であることを確認。

### 9.2 環境上の一過性の異常(コードバグではない): ticker 5248の
Score Parquetファイルで`PermissionError`

- 現象: `run_build_scores()`がticker 5248のScoreファイルを正常に書き込んだ
  直後(同一実行内)、`run_walk_forward()`内の`build_scored_with_targets()`
  がそのファイルを読み込もうとした際に一度だけ`PermissionError: [Errno 13]`
  が発生。
- 調査: 実行完了後に同じファイルを独立したPythonプロセスから読み込んだ
  ところ、正常に読み込め、データも正しく642行含まれていた。Windows環境
  特有の一過性のファイルロック(ウイルススキャン等による書き込み直後の
  排他ロック)が疑われる、再現性のない環境要因であり、コード上の論理
  バグではないと判断。
- 影響評価: 既存の`try/except Exception`により1銘柄分がログに記録されて
  スキップされただけで、実行全体は正常完了。Signal/Backtest/Trade関連の
  結果(§2の表)には一切影響しない(このティッカーは`run_backtest`では
  正常に処理されている)。影響が及ぶのは Score Validation(Q1-Q5)の母集団
  からこの1銘柄が漏れた点のみで、2,880銘柄中の1銘柄であり無視できる
  水準。
- 対応: コード修正は行っていない(再現しない環境要因のため、事後対応が
  不要と判断)。

---

## 10. Final Case Classification

| Case | 定義 | 該当Signal数 | 銘柄 |
|---|---|---:|---|
| A | 両フェーズでPF<1、優位性を支持しない | 10 | long_breakout, long_ma_rebound, long_momentum_continuation, long_volume_breakout, short_breakdown, short_ma_rejection, short_momentum_continuation, short_overbought_reversal, short_pullback, short_volume_breakdown |
| B | Phase 6.5でPF<1→Phase 7でPF>1(regime依存調査対象) | 0 | (該当なし) |
| C | Phase 6.5でPF>1→Phase 7で再現せず | 1 | long_pullback |
| D | 両フェーズでPF>1、本格的候補として扱う(自動採用は禁止) | 1 | long_oversold_rebound |

---

## 結論

Phase 7は「Phase 6.5の結果が未知の未来期間でも再現するか」を検証する
という当初の目的を達成した。12 Signal中10がCase A(両フェーズで一貫して
無効)という結果は、Phase 6.5のREJECT判定が単なる期間固有のノイズでは
なかったことの追加的な証拠となる。

唯一の例外である`long_oversold_rebound`(Case D)は、両フェーズでPF>1・
Bootstrap CIがゼロを跨がない・Permutation pが有意・FDR補正後も有意
(Phase 7では特に強い)・複数WFO Window(5/9)で再現・2,000以上の銘柄に
分散(集中度は低い)という、仕様が定めるCase Dの実質的な条件を満たして
いる。**Decision Framework(不変)によりACCEPT_CANDIDATEと機械的に分類
された**。

しかし、Regime別分析(§4)により、この結果はほぼ全面的にBEAR regime
(60日リターン<-5%の58営業日、PF=42.8)に依存しており、BULL/NEUTRAL
regimeでは優位性がほぼ消失することが判明した。これは「常に効くSignal」
ではなく「相場急落局面に限定された押し目買いの優位性」という、より
限定的だが実務的には無視できない発見である。

**この時点で実運用への自動採用は一切行わない。** 仕様が定める通り、
Phase 6.5・Phase 7の両方で再現性を確認した後の人間による採用判断が
必要であり、特にBEAR regimeへの依存度の高さ、BEAR判定58営業日の内訳
(特定の急落イベントへの集中の有無)については、本レポートの範囲を
超えるさらなる人間による検証を推奨する。

**Phase 7完了。Phase 8以降には進まない。**

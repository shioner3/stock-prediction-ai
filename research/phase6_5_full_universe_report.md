# Phase 6.5: Full Universe OOS Validation 最終レポート

実行日: 2026-08-20
config_hash: `a9b34ccb6e6e1e1a9d4d1eb554fe3fd8a4c09b2d9d186b2bd0ee3b0d430a2217`
data_hash: `8d824e03e2428e7587830a86879dba64922adc6e5f69a1070f8d8000b04dc843`

## 0. 結論の要約

12 Signal中 **11がREJECT、1がINSUFFICIENT_EVIDENCE、0がACCEPT_CANDIDATE**。

これは「4銘柄だから検出できなかった」ではなく「~2,700銘柄・約39万トレードの
横断的サンプルでも優位性を統計的に確認できなかった」という結果である。
Concentration分析(§6)が示す通り、どのSignalも上位10銘柄で取引数の3%未満
しか占めておらず、結果が少数銘柄に依存した見かけ上の優位性ではないことを
確認済み。0 ACCEPT_CANDIDATEは本検証における妥当な結果であり、これを理由に
新規Signalを追加することはしない(仕様のCase G)。

唯一INSUFFICIENT_EVIDENCEとなった`LONG:long_oversold_rebound`も、Base costでは
PF 1.16・bootstrap CI下限>0・permutation p=0.0055と好調に見えるが、(a) High
cost tierでPFが0.89に落ち込み実coverageに耐えない、(b) 12 Signal分の多重検定
補正(FDR)後は adjusted p=0.066となり従来の有意水準0.05を割り込む。単独の
raw p-valueだけで有望と判断しないための多重検定チェックが機能した一例。

## 1. Dataset Summary

| 項目 | 値 |
|---|---|
| JPX Master候補数 | 4,444 |
| 内訳 | Prime 1,558 / Standard 1,559 / Growth 596 / ETF・ETN 476 / PRO Market 185 / REIT 63 / 外国株 5 / その他 2 |
| Static Filter通過(Prime+Standard+Growth) | 3,713 |
| Static Filter除外 | 731 |
| Fetch試行 | 3,713 |
| Fetch成功 / 部分成功 / 失敗 | 3,378 / 201 / 134 |
| Price/Liquidity Filter除外 | 824 |
| **Final Universe(Signal評価対象)** | **2,755銘柄** |
| データ期間(base case, Phase 6と同一) | 2022-01-04 〜 2024-06-28 |
| Walk Forward Window数 | 5(Phase 6と完全一致、window数変更なし) |
| 総トレード数(全12 Signal合算) | 389,192 |
| Survivorship Bias | **警告あり**(§8参照。無料公式データでは過去の上場廃止銘柄を再構成できないため、本検証は"Current Universe"方式) |

Walk Forward Windows(Phase 6のREADME記載の5 Windowと日付が完全一致 - 同一base case・同一WFO設定から機械的に導出されるため当然の結果):

| Window | TRAIN | VALIDATION | OOS |
|---|---|---|---|
| 0 | 2022-01-04〜2023-01-04 | 2023-01-04〜2023-04-04 | 2023-04-04〜2023-07-04 |
| 1 | 2022-04-04〜2023-04-04 | 2023-04-04〜2023-07-04 | 2023-07-04〜2023-10-04 |
| 2 | 2022-07-04〜2023-07-04 | 2023-07-04〜2023-10-04 | 2023-10-04〜2024-01-04 |
| 3 | 2022-10-04〜2023-10-04 | 2023-10-04〜2024-01-04 | 2024-01-04〜2024-04-04 |
| 4 | 2023-01-04〜2024-01-04 | 2024-01-04〜2024-04-04 | 2024-04-04〜2024-06-28(短縮) |

## 2. Per-Signal Summary Table(OOS, Base cost tier)

| Signal | Dir | Trades | Unique銘柄 | Win Rate | PF(base) | Expectancy | Bootstrap CI(expectancy) | Perm. p | Windows PF>1 | Decision |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| long_breakout | LONG | 18,749 | 2,724 | 43.1% | 0.855 | -0.00328 | [-0.00429, -0.00230] | 1.000 | 0/5 | REJECT |
| long_ma_rebound | LONG | 17,414 | 2,682 | 47.0% | 0.961 | -0.00066 | [-0.00146, +0.00014] | 0.799 | 2/5 | REJECT |
| long_momentum_continuation | LONG | 38,982 | 2,735 | 44.4% | 0.883 | -0.00244 | [-0.00306, -0.00181] | 1.000 | 0/5 | REJECT |
| long_oversold_rebound | LONG | 2,997 | 1,370 | 50.9% | 1.160 | +0.00285 | [+0.00074, +0.00502] | 0.0055 | 4/5 | INSUFFICIENT_EVIDENCE |
| long_pullback | LONG | 20,851 | 2,613 | 47.5% | 1.019 | +0.00039 | [-0.00047, +0.00125] | 0.0657 | 2/5 | REJECT |
| long_volume_breakout | LONG | 11,206 | 2,586 | 42.4% | 0.871 | -0.00360 | [-0.00513, -0.00200] | 1.000 | 2/5 | REJECT |
| short_breakdown | SHORT | 13,152 | 2,699 | 42.4% | 0.649 | -0.00754 | [-0.00845, -0.00663] | 0.0248 | 0/5 | REJECT |
| short_ma_rejection | SHORT | 15,901 | 2,685 | 43.7% | 0.680 | -0.00602 | [-0.00684, -0.00522] | 0.4821 | 0/5 | REJECT |
| short_momentum_continuation | SHORT | 34,521 | 2,738 | 42.6% | 0.632 | -0.00829 | [-0.00889, -0.00769] | 0.1328 | 0/5 | REJECT |
| short_overbought_reversal | SHORT | 6,529 | 2,173 | 46.0% | 0.723 | -0.00620 | [-0.00776, -0.00474] | 0.1166 | 0/5 | REJECT |
| short_pullback | SHORT | 18,487 | 2,540 | 47.6% | 0.759 | -0.00525 | [-0.00610, -0.00441] | 0.9899 | 0/5 | REJECT |
| short_volume_breakdown | SHORT | 8,528 | 2,550 | 48.5% | 0.812 | -0.00473 | [-0.00627, -0.00320] | 0.6947 | 0/5 | REJECT |

## 3. Cost Sensitivity Table(Profit Factor, 4 tiers)

| Signal | Zero(0bps) | Low(10bps) | Base(30bps) | High(80bps) |
|---|---:|---:|---:|---:|
| long_breakout | 0.987 | 0.940 | 0.855 | 0.676 |
| long_ma_rebound | 1.150 | 1.083 | 0.961 | 0.715 |
| long_momentum_continuation | 1.029 | 0.978 | 0.883 | 0.687 |
| long_oversold_rebound | 1.358 | 1.288 | 1.160 | 0.895 |
| long_pullback | 1.180 | 1.124 | 1.019 | 0.800 |
| long_volume_breakout | 0.977 | 0.940 | 0.871 | 0.721 |
| short_breakdown | 0.771 | 0.728 | 0.649 | 0.488 |
| short_ma_rejection | 0.824 | 0.773 | 0.680 | 0.495 |
| short_momentum_continuation | 0.746 | 0.706 | 0.632 | 0.479 |
| short_overbought_reversal | 0.846 | 0.803 | 0.723 | 0.556 |
| short_pullback | 0.889 | 0.843 | 0.759 | 0.582 |
| short_volume_breakdown | 0.927 | 0.887 | 0.812 | 0.651 |

コスト無しでもPF>1のSignalは4つ(long_ma_rebound, long_oversold_rebound,
long_pullback)のみ。SHORT系は全てZero-cost tierでもPF<1であり、コスト以前
に基礎的な優位性が確認できない。

## 4. Regime × Signal Table(Win Rate / Expectancy)

TOPIX Proxy 60日トレーリングリターンで分類。**この検証期間(2022-01〜
2024-06)中、BEAR判定(60日リターン<-5%)となった日が一度も無かったため、
BEAR regimeのデータは0件。** Signal のBear相場での挙動は本データセットでは
検証できていない(既知の限界、Phase 7以降でデータ期間拡張時に再評価が必要)。

| Signal | BULL n / WinRate / Exp | NEUTRAL n / WinRate / Exp |
|---|---|---|
| long_breakout | 12,720 / 44.8% / -0.00019 | 6,029 / 47.3% / -0.00048 |
| long_ma_rebound | 11,521 / 49.5% / +0.00227 | 5,893 / 51.4% / +0.00247 |
| long_momentum_continuation | 26,297 / 46.1% / +0.00005 | 12,685 / 48.9% / +0.00162 |
| long_oversold_rebound | 1,371 / 48.9% / +0.00057 | 1,626 / 57.2% / +0.01030 |
| long_pullback | 14,150 / 49.2% / +0.00347 | 6,701 / 52.0% / +0.00321 |
| long_volume_breakout | 7,503 / 44.1% / -0.00011 | 3,703 / 44.6% / -0.00160 |
| short_breakdown | 7,010 / 47.8% / -0.00177 | 6,142 / 41.3% / -0.00770 |
| short_ma_rejection | 9,054 / 47.4% / -0.00216 | 6,847 / 45.3% / -0.00416 |
| short_momentum_continuation | 18,540 / 46.4% / -0.00408 | 15,981 / 43.7% / -0.00669 |
| short_overbought_reversal | 4,567 / 50.5% / -0.00278 | 1,962 / 46.2% / -0.00417 |
| short_pullback | 9,546 / 51.8% / -0.00048 | 8,941 / 48.1% / -0.00413 |
| short_volume_breakdown | 5,322 / 52.7% / -0.00019 | 3,206 / 46.8% / -0.00428 |

`long_oversold_rebound`はNEUTRAL regimeでのみ強い(Expectancy +0.0103)。BULLでは
ほぼフラット(+0.0006)。「特定regimeでのみ効く」パターンの典型例(仕様の
分類でいうCase B寄り)。

## 5. Score Q1-Q5 Validation(quantile buckets, forward 5d, 抜粋)

Quantile bucketは**OOS全体の総合Score分布**から算出しているため、Score帯が
狭いSignalは単一bucketに収束する場合がある(例: long_oversold_rebound自身の
トリガー行は全てQ1相当に収まり、単独ではQ1-Q5比較が成立しない)。

複数bucketに分散した`long_pullback`(forward 5d, n=57,422)の例:

| Bucket | n | Avg Return |
|---|---:|---:|
| Q1(最低Score) | 21,603 | 0.00404 |
| Q2 | 19,037 | 0.00561 |
| Q3 | 8,630 | 0.00598 |
| Q4 | 5,869 | 0.00477 |
| Q5(最高Score) | 1,424 | 0.00409 |

- Monotonic: **False**(単調性なし。中間帯Q3がピーク)
- Rank相関(ordinal rank vs avg return): **-0.133**(弱い負の相関)
- Q5-Q1 spread: +0.0000499(ほぼゼロ)
- Q5-Q1 bootstrap CI: [-0.00423, +0.00452](ゼロを含む → 有意差なし)

**Scoreの単調性は、少なくともこの2つのSignalでは確認できなかった。**
Score自体の重み・閾値は本Phaseでは一切変更していない(仕様の絶対遵守事項)。

## 6. Concentration Table(cross-sectional)

| Signal | Unique銘柄 | Top1 trade share | Top5 | Top10 | Top1 return share | Top10 |
|---|---:|---:|---:|---:|---:|---:|
| long_breakout | 2,724 | 0.10% | 0.43% | 0.82% | -37.8% | -268% |
| long_ma_rebound | 2,682 | 0.10% | 0.48% | 0.91% | +2.8% | +21.1% |
| long_momentum_continuation | 2,735 | 0.08% | 0.39% | 0.77% | +10.8% | +76.3% |
| long_oversold_rebound | 1,370 | 0.47% | 1.77% | 3.24% | +5.4% | +28.3% |
| long_pullback | 2,613 | 0.15% | 0.68% | 1.26% | +3.2% | +20.9% |
| long_volume_breakout | 2,586 | 0.12% | 0.60% | 1.15% | -27.0% | -174% |
| short_breakdown | 2,699 | 0.11% | 0.53% | 1.02% | -1.3% | -8.7% |
| short_ma_rejection | 2,685 | 0.13% | 0.58% | 1.08% | -1.4% | -10.2% |
| short_momentum_continuation | 2,738 | 0.09% | 0.44% | 0.86% | -0.6% | -3.9% |
| short_overbought_reversal | 2,173 | 0.18% | 0.87% | 1.70% | -2.4% | -18.3% |
| short_pullback | 2,540 | 0.16% | 0.74% | 1.43% | -3.0% | -16.7% |
| short_volume_breakdown | 2,550 | 0.12% | 0.59% | 1.17% | -5.3% | -36.0% |

Trade share(取引数ベース)は全Signalでtop10銘柄が1.3~3.2%程度に収まって
おり、極端な集中は無い。Return share(損益寄与ベース)が100%を超える/大きく
負になっているケースがあるのは、PFが1未満のSignal(全体としては損失)で
少数の勝ち銘柄の利益を多数の負け銘柄の損失が相殺・逆転しているため(モジュール
docstring記載の通り、意図的にクリップしていない生の値)。「少数銘柄依存の
見せかけの優位性」という懸念は本結果からは支持されない。

## 7. Data Quality Table

| 項目 | 値 |
|---|---|
| 対象銘柄(raw, 取得成功+部分成功) | 3,579 |
| 総行数(生OHLCV合計) | 2,112,256 |
| 重複日付行 | 0 |
| 不正OHLC行(価格≤0 or high<low) | 0 |
| 負のVolume行 | 0 |
| ゼロVolume行 | 25,852(全行の約1.2%、低流動性日として許容範囲) |
| NaN行 | 3(無視できる水準) |
| Coverage比率(対期待営業日数) 平均 / 中央値 | 90.9% / 94.0% |
| Coverage比率<50%の銘柄数 | 123(主に2024年新規上場のalphanumericコード銘柄) |

## 8. Integrity / Hash / Survivorship Bias

- `config_hash` = `a9b34ccb6e6e1e1a9d4d1eb554fe3fd8a4c09b2d9d186b2bd0ee3b0d430a2217`
  (config/settings.yaml + config/universe_filters.yaml のsha256結合ハッシュ)
- `data_hash` = `8d824e03e2428e7587830a86879dba64922adc6e5f69a1070f8d8000b04dc843`
  (Full Universe 2,755銘柄分のFeature Parquetファイル群のsha256結合ハッシュ)
- **Survivorship Bias警告: 有効**。無料の公式JPXデータでは過去の上場廃止銘柄・
  上場日を再構成できないため、本検証は"Current Universe"(2026年8月時点で
  上場している銘柄を2022-2024年に遡って評価)方式である。J-Quants DataCube
  等の有料/登録制サービスへの登録はユーザーの明示的許可を要するため本Phase
  では行っていない(ユーザーへの確認により明示的に選択された方針)。この
  結果は「2022-2024年に存在した全銘柄」ではなく「現在も存在する銘柄の
  2022-2024年時点の値動き」に基づく点に注意。
- Universe構築時の新規look-ahead監査: `universe/filters.py`の流動性フィルタが
  fetch範囲の**末尾**(直近)データを使って2022年時点のUniverse組入れを判定
  していたバグを発見・修正済み(`.tail()` → `.head()`、"Phase 6.5 Data/
  Universe Leakage Fix")。Full Universe検証はこの修正後のコードで実行。
- 再現性: 同一config・同一データで`run_walk_forward()`を複数回実行しても
  Signal数・Trade数・PF・Expectancy・Bootstrap CI・Permutation p値・Decision
  が完全一致することを`tests/test_pipeline_run_walk_forward.py`で確認済み
  (Full Universe固有のテストとして`tests/test_walk_forward_no_lookahead.py`
  にバッチ処理の非汚染性テストも追加)。

## 9. 本Phaseで見つかり修正したバグ

Full Universeスケールで初めて顕在化した実バグを3件発見・修正した(いずれも
Signal/Score/Backtestの計算ロジック自体には手を入れていない):

1. **Universe構築時のlook-ahead**(`universe/filters.py`): 流動性フィルタが
   fetch範囲の最新データで過去のUniverse組入れを判定していた。`.head()`化で
   修正。
2. **CA証明書パス問題**(環境要因): プロジェクトパスに日本語が含まれるため
   curl_cffi(yfinanceが内部で使用)がCA証明書を読み込めず、Full Universe
   取得が全銘柄で失敗していた。ASCIIパスにコピーした`cacert.pem`を
   `CURL_CA_BUNDLE`/`SSL_CERT_FILE`で指定して解消(コード変更なし、実行時
   環境変数のみ)。
3. **Permutation Testのメモリスケーリング**(`backtest/permutation.py`):
   Full Universeの母集団サイズ(約81万行)×10,000回の順列を一括配列化すると
   約64GBのメモリを要求しクラッシュしていた。統計的に同一の結果を保つ形で
   チャンク分割処理に変更(rng消費順序が同一であることをテストで直接検証
   済み)。
4. (軽微)**ゼロSignalティッカーのエラーログ**(`pipeline/run_score_validation.py`):
   一度もSignalが発火しなかった銘柄(Scoreファイルが存在しない)を、想定外の
   例外としてERRORログに記録していた。Full Universeでは日常的に起きるため、
   正常系として静かにスキップするよう修正。

## 10. Case分類(仕様のCase A〜G)

**Case G「12 SignalすべてREJECT」に該当**(long_oversold_reboundのみ
INSUFFICIENT_EVIDENCEだが、これもACCEPT_CANDIDATEではない)。仕様の指示通り、
これを理由に新規Signalを追加する対応は行わない。数値上唯一目を引いた
long_oversold_reboundも、多重検定補正・高コスト耐性の両方でACCEPT水準に
届いておらず、Case C(コスト耐性なし)寄りの限定的な結果として記録するに
留める。

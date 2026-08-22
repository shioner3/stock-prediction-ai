# PHASE 9 REPORT: long_oversold_rebound 独立ロバストネス検証

実行日: 2026-08-20
config_hash: `a9b34ccb6e6e1e1a9d4d1eb554fe3fd8a4c09b2d9d186b2bd0ee3b0d430a2217`
(Phase 6.5・Phase 7・現在の3値が完全一致、CONFIG_MISMATCHなし)

**本Phaseの目的は有効性を証明することではなく、有効性が崩れる条件を
積極的に探すことだった。** 結論から言うと、崩れる条件は複数見つかった
(BEAR regime内の日次集中度は想定以上に極端、timing offsetを数営業日
ずらすだけで優位性は急速に消失する)一方で、崩れない側面も複数見つかった
(2024年を丸ごと除いてもCombined全体のPFは1を割らない、6つの事前固定
シナリオ全てでPF>1を維持、2024年8月episode除外後もBEAR限定でHigh cost
PF=10超を維持)。

---

## 0. 最終分類(先出し)

> **Primary: EVENT_DEPENDENT**
> **Secondary caveat: REGIME_DEPENDENT**

BEAR regime限定の優位性は、たった1営業日で全BEAR損益の64.8%、3営業日で
93.7%を占めるという極端な集中を示し(Gini係数0.956)、timing offset sweep
でも実際のSignalタイミングから数営業日ずれるだけで優位性が急速に失われる
(offset=-15で早くもPF=0.095に崩壊)。これは「BEAR regimeという条件全般で
安定的に再現する現象」というより「2024年8月の特定イベント(とその直近数日)
に強く紐づいた現象」であることを示す。**Primary分類をEVENT_DEPENDENTとする。**

一方で、そのイベントを完全に除外してもBEAR限定でHigh cost PF=10.01を
維持し、Day Cluster/Block Bootstrap(取引日単位・5営業日ブロック単位の
より保守的な再抽出)でもBEAR PFの信頼区間下限は1を大きく上回ったまま
(4.32、2.80)であり、単なる偶然の1イベントだけでは説明しきれない
regime条件付きの残存エッジも確認された。**Secondary caveatとして
REGIME_DEPENDENTを付記する。**

実運用への自動採用は行っていない。

---

## 1. Dataset

| 項目 | 値 |
|---|---|
| データ範囲 | 2022-01-04〜2026-08-20(`data/phase7/`を再利用、新規取得なし) |
| Universe | 2,880銘柄(Phase 7・8と同一) |
| long_oversold_reboundトレード総数 | 12,014(全regime、Combined) |
| BEAR regimeトレード数 | 1,226 |
| BEAR episode数 | 13(Phase 8と完全一致) |
| Survivorship Bias | Phase 6.5/7/8と同一のCurrent Universe警告を維持 |

---

## 2. Episode Analysis(拡張版、Unique銘柄数・中央値・最大DDを追加)

| # | 期間 | Trades | Unique銘柄 | PF | 中央値Return | Max DD(episode内) | P&L寄与 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | 2022-04-11〜04-12 | 11 | 11 | 2.09 | +0.24% | -2.07% | 0.05% |
| 1 | 2022-06-21〜06-23 | 65 | 65 | 12.68 | +4.00% | -10.83% | 2.52% |
| 2 | 2022-07-01 | 0 | 0 | - | - | - | - |
| **3** | **2024-08-02〜08-15** | **652** | **647** | **130.96** | **+11.05%** | **-16.73%** | **71.62%** |
| 4 | 2024-08-19 | 1 | 1 | ∞ | +0.27% | 0.00% | 0.00% |
| 5 | 2024-08-21〜08-22 | 2 | 2 | 0.09 | -1.64% | -3.62% | -0.03% |
| 6 | 2024-08-27〜08-29 | 0 | 0 | - | - | - | - |
| 7 | 2024-09-04〜09-26 | 53 | 46 | 2.17 | +0.98% | -35.62% | 0.73% |
| 8 | 2024-09-30〜10-04 | 10 | 10 | 1.78 | -1.46% | -12.46% | 0.13% |
| 9 | 2024-10-08〜10-16 | 19 | 19 | 0.23 | -2.60% | -49.70% | -0.42% |
| 10 | 2025-04-03〜04-17 | 398 | 394 | 78.59 | +6.24% | -5.33% | 25.16% |
| 11 | 2025-04-21〜04-24 | 1 | 1 | ∞ | +8.54% | 0.00% | 0.08% |
| 12 | 2026-03-30〜03-31 | 14 | 14 | 3.01 | +0.47% | -8.82% | 0.18% |

Episode 3と10だけでP&L寄与の96.8%(71.62%+25.16%)を占める。**BEAR
regime内の優位性は実質2つのepisodeに支えられている。**

---

## 3. Leave-One-BEAR-Episode-Out(Bootstrap CI付き)

| 除外Episode | 除外Trades | Full PF | 除外後PF | 除外後Expectancy 95% CI |
|---|---:|---:|---:|---|
| 0 | 11 | 39.57 | 40.27 | [0.0890, 0.0983] |
| 1 | 65 | 39.57 | 42.01 | [0.0907, 0.1002] |
| **3** | **652** | **39.57** | **14.90** | **[0.0514, 0.0611]** |
| 4 | 1 | 39.57 | 39.57 | [0.0883, 0.0974] |
| 5 | 2 | 39.57 | 40.06 | [0.0883, 0.0976] |
| 7 | 53 | 39.57 | 51.31 | [0.0916, 0.1009] |
| 8 | 10 | 39.57 | 42.09 | [0.0887, 0.0981] |
| 9 | 19 | 39.57 | 50.22 | [0.0900, 0.0991] |
| 10 | 398 | 39.57 | 33.99 | [0.0967, 0.1089] |
| 11 | 1 | 39.57 | 39.54 | [0.0881, 0.0974] |
| 12 | 14 | 39.57 | 40.87 | [0.0891, 0.0984] |

Episode 3除外だけがExpectancy信頼区間を明確に押し下げる
(通常[0.088〜0.101]帯 → [0.051, 0.061])。ただしこの除外後CIも**ゼロを
明確に上回っており**、Episode 3を除いても統計的な優位性自体は消えない。

---

## 4. Leave-One-Year-Out(Combined、全regime対象)

| 除外年 | 除外Trades | Full PF | 除外後PF |
|---|---:|---:|---:|
| 2022 | 3,009 | 1.865 | 1.825 |
| 2023 | 2,395 | 1.865 | 1.935 |
| **2024** | **2,644** | **1.865** | **1.577** |
| 2025 | 2,240 | 1.865 | 1.854 |
| 2026 | 1,726 | 1.865 | 2.123 |

**2024年を丸ごと除外してもCombined全体のPFは1.577で、1を大きく上回った
まま。** これはScenario C(§9)と完全に一致する検算結果でもある。特定の
1年だけに依存した結果ではないことを示す、ポジティブな材料。

---

## 5. Day-Level Event Concentration

| Top-K日 | P&L寄与 | Trade寄与 |
|---|---:|---:|
| 1 | **64.8%** | 45.4% |
| 3 | 93.7% | 80.2% |
| 5 | 96.7% | 85.9% |
| 10 | 98.9% | 90.5% |
| 20 | 100.4% | 95.4% |

Gini係数(日次P&L): **0.956**(1に近いほど極端な集中)。

「9営業日で71.6%」というPhase 8の所見は、より細かく見ると**「たった1
営業日で64.8%」**というさらに極端な集中だったことが判明した。この1日は
おそらく2024年8月急落の翌営業日(急反発が最も鋭く出た日)に対応すると
推定される。

---

## 6. Bootstrap: 3方式比較(Trade-level / Day Cluster / Block)

### BEAR regime限定

| 方式 | PF点推定 | PF 95% CI | Expectancy 95% CI |
|---|---:|---|---|
| Trade-level(Phase 8) | 34.89 | [26.76, 47.07] | [0.0852, 0.0943] |
| **Day Cluster**(Phase 9) | 39.57 | **[4.32, 87.32]** | [0.0301, 0.1179] |
| **Block(5営業日)**(Phase 9) | 39.57 | **[2.80, 87.68]** | [0.0217, 0.1179] |

### Combined(全regime)

| 方式 | PF点推定 | PF 95% CI |
|---|---:|---|
| Day Cluster | 1.865 | [1.20, 2.85] |
| Block(5営業日) | 1.865 | [1.10, 2.93] |

**この比較こそが本Phaseの核心的所見の一つ。** Trade-levelのCIは
[26.76, 47.07]と非常にタイトで「BEAR PFは34.9±10程度」という誤った
確信を与えるが、日次クラスタ・時間ブロックを考慮すると信頼区間は
桁違いに広がる([2.80〜87.68])。**Phase 8で使用したTrade-level
Bootstrapは、BEAR regime限定の優位性について不確実性を大幅に過小評価
していたことが直接的に確認された。**

ただし重要な点として、CIの下限([4.32]、[2.80])は3方式いずれも依然として
1を大きく上回っている。「不確実性は非常に大きいが、ゼロ(PF=1)である
可能性は低い」という、より正確でニュアンスのある結論になる。Combined側
のCI下限([1.20]、[1.10])もいずれも1を上回っており、全regime込みの
優位性についてはより頑健な結論が得られる。

---

## 7. Timing Placebo Sweep(事前固定offset: -15,-10,-5,-3,-1,+5,+10)

| Offset(営業日) | BEAR Trades | PF | Expectancy |
|---:|---:|---:|---:|
| -15 | 238 | 0.095 | -0.0305 |
| -10 | 229 | 0.098 | -0.0419 |
| -5 | 175 | 0.036 | -0.0677 |
| -3 | 617 | 0.173 | -0.0359 |
| **-1** | 1,233 | **20.64** | +0.0764 |
| **+5** | 1,279 | **5.02** | +0.0285 |
| **+10** | 1,276 | **4.14** | +0.0239 |
| (実際のSignal, 参考) | 1,226 | **39.57** | +0.0927 |

**明確な時間的局所性が確認された。** -15〜-3営業日のoffsetでは優位性が
ほぼ完全に消失(PF<1、多くはPF<0.2)する一方、-1営業日・+5・+10営業日
というより近いoffsetでは、実際のSignalには及ばないもののなお顕著な
PF(4〜21倍)が残る。

この「近傍だけ効く」パターンは2通りに解釈できる:

1. **好意的な解釈**: Signal自体のタイミング(RSI<30+陽転日)に本当に
   意味があり、1日ずらすだけでも精度が急落する高精度なタイミング
   シグナルである。
2. **懐疑的な解釈**: 2024年8月急落・反発イベントの期間自体が数営業日の
   幅を持つため、その期間内であればどこを取っても「たまたま」大きな
   リターンを拾ってしまう(イベントウィンドウ内での時間的自己相関)。

本レポートはこの2つを完全に切り分けることはできない。**両方の力学が
同時に働いている可能性が高い**というのが最も正直な結論である。

---

## 8. Cross-Sectional Robustness

### Sector(33業種区分、JPX Master由来、BEAR regime限定)

31業種全てでBEAR PFがプラス(多くがPF=inf、極小サンプルで負けトレードが
皆無なだけ)。取引数上位: 情報・通信業(n=250, PF=41.0)、機械(n=106,
PF=73.4)、電気機器(n=144, PF=55.1)、サービス業(n=188, PF=62.5)。
**特定業種への依存は確認されない。**

### Liquidity(既存Universe filterと同じ基準、4分位)

| Bucket | Trades | PF |
|---|---:|---:|
| Q1(低流動性) | 307 | 24.49 |
| Q2 | 306 | 93.75 |
| Q3 | 306 | 39.50 |
| Q4(高流動性) | 307 | 38.17 |

流動性による単調な劣化は見られず、**低流動性銘柄だけが結果を作っている
わけではない**(Q1が最も低いが、それでもPF=24超)。

### Market-cap proxy

`NOT_AVAILABLE`(本プロジェクトは発行済株式数を取得しておらず、時価総額
proxyを構成できないため。情報がない場合は実施しないという仕様section 12
の指示に従い、無理な代替指標は導入していない)。

---

## 9. Cost Stress(4 Tier × 4 Scope)

| Scope | Zero | Low | Base | High |
|---|---:|---:|---:|---:|
| Combined(全regime) | 1.865 | 1.781 | 1.626 | 1.297 |
| BEAR全体 | 39.57 | 37.96 | 34.89 | 28.07 |
| 2024-08 episodeのみ | 130.96 | 127.10 | 119.82 | 100.69 |
| **BEAR(2024-08除外)** | 14.90 | 14.19 | 12.85 | **10.01** |

Combined全体は最もコストが厳しいHigh tierでもPF>1を維持(1.297)。
**2024-08 episodeを完全に除いたBEAR限定でも、High costでPF=10.01を維持** -
この単一の数字が、Secondary caveatとしてREGIME_DEPENDENTを併記した
最大の根拠である。

---

## 10. Forward Holding Period Sensitivity(研究専用、Backtest結果は変更せず)

| Horizon | n | Mean | Median | Win Rate | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---|
| 1d | 20,666 | 0.213% | 0.00% | 49.7% | [0.172%, 0.256%] |
| 3d | 20,663 | 0.831% | 0.41% | 54.7% | [0.759%, 0.904%] |
| 5d | 20,656 | 1.241% | 0.71% | 56.5% | [1.145%, 1.338%] |
| 7d | 20,644 | 1.518% | 0.85% | 56.7% | [1.405%, 1.634%] |
| 10d | 20,623 | 1.684% | 0.81% | 55.7% | [1.551%, 1.817%] |

平均リターン・勝率とも1日目から段階的に上昇し、5〜7日目あたりで勝率が
ピークを迎える。**「1日だけの跳ね返りを拾っているだけ」という懸念は
支持されない** - リターンは複数日にわたって緩やかに積み上がっており、
既存のHOLD_DAYS=5は、この寄与カーブの初期の大部分を捉えている合理的な
選択と言える(Backtest仕様自体は本Phaseで変更していない)。

---

## 11. Predefined Stress Scenarios(A〜F、事前固定)

| Scenario | 説明 | Trades | PF |
|---|---|---:|---:|
| A | Full Combined OOS(無調整) | 12,014 | 1.865 |
| B | 2024-08-02〜08-15除外 | 11,362 | 1.444 |
| C | 2024年全体除外 | 9,370 | 1.577 |
| D | 各BEAR episode leave-one-out | (§3参照) | (§3参照) |
| E | Top/Bottom 1% winsorization | 12,014 | 1.862 |
| F | E + B(winsorization + 2024-08除外) | 11,362 | 1.422 |

**Scenario A〜Fの全てでPF>1を維持**(D=各episode除外時もPFは常に1を
上回る、§3参照)。最も厳しい組み合わせであるFでもPF=1.422。Combined
(全regime込み)レベルで見る限り、この結果はいずれの事前固定ストレス
テストでも崩壊しない。

winsorization(E)がAとほぼ同じ(1.862 vs 1.865)なのは、Combined全体
(12,014トレード)に対して極端値の影響が相対的に薄まるため。BEAR regime
限定で見た場合の極端な集中度(§5)とは対照的な結果であり、「全体としては
頑健、regime条件付きで見ると非常に脆い」という本レポート全体の結論と
整合する。

---

## 12. Q1〜Q8への回答(Phase 8由来、Phase 9で更新)

新たな知見のみ要約(Phase 8時点の回答は`research/phase8_report.md`参照):

- **Q3(複数期間での再現)**: Day-level集中度分析により、実質的には
  「複数episodeというより、ほぼ1営業日+2つ目のepisode」という、Phase 8
  時点よりもさらに集中した実態が判明した。
- **Q7(統計的有意性)**: Day Cluster/Block Bootstrapという、より保守的な
  再抽出方式でもBEAR PFの信頼区間下限は1を上回ったままだが、区間の
  「幅」自体はTrade-level bootstrapの想定よりも遥かに広く、Phase 8で
  示唆した「Permutation Testの独立性仮定の破れ」という懸念が、独立した
  bootstrap手法によっても定量的に裏付けられた。
- **Q8(episode除外の影響)**: 変わらず「消えないが大幅に縮小する」。
  Timing Offset Sweepの結果と合わせて考えると、この縮小はもはや
  「1つのepisodeへの依存」という単純な話ではなく、「その前後数営業日
  というごく狭い時間窓への依存」という、より鋭い形の依存性であること
  が明らかになった。

---

## 13. Bugs

Phase 9で新たに発見したコード上のバグは**なし**。Phase 8で確認済みの
「TOPIX Proxy取得漏れ」バグは既に修正済みであり、Phase 9はこの修正後の
コードをそのまま再利用した(config_hash一致・実行時のTOPIXデータ正常
読込により再確認済み)。

---

## 14. Integrity

- `config_hash`(現在・Phase 6.5・Phase 7) = `a9b34ccb6e6e1e1a9d4d1eb554fe3fd8a4c09b2d9d186b2bd0ee3b0d430a2217`
  (3値完全一致、CONFIG_MISMATCHなし)
- Phase 9専用パラメータ(`config/phase9_settings.yaml`、Signal/Score/
  Backtest/WFOには一切影響しない別ファイル): block_length_days=5、
  day cluster bootstrap n_resamples=10,000/seed=44、block bootstrap
  n_resamples=10,000/seed=45、timing offsets=[-15,-10,-5,-3,-1,5,10]、
  winsorization=[1st, 99th] percentile。全て実行前に固定、結果を見て
  からの変更は一切なし。
- データ: `data/phase7/`を無変更で再利用(新規Fetchなし)。Phase 6.5/7/8
  の既存データ・レポートは一切上書きしていない。
- テスト: pytest 558 passed / 2 deselected(Phase 8の519から+39、新規
  Phase 9テスト約40件を含む)、ruff/mypyともクリーン。
- 依存方向テスト(section 19): `tests/test_phase9_no_lookahead.py`で
  `features/`・`signals/`が`pipeline.run_phase8_analysis`・
  `pipeline.run_phase9_analysis`・Phase 8/9の新規`backtest/`分析モジュール
  のいずれも一切importしていないことをAST静的解析で機械的に確認
  (`tests/test_target_leakage.py`の既存パターンを踏襲)。
- 再現性テスト: 同一config・同一データ・同一seedで`run_phase9_analysis()`
  を2回実行し、episode一覧・LOPO結果・timing placebo・scenario結果が
  完全一致することを`tests/test_pipeline_run_phase9_analysis.py`で
  直接検証。
- 既存モジュール(`backtest/decision.py`, `backtest/bootstrap.py`,
  `backtest/permutation.py`, `backtest/costs.py`, `backtest/market_regime.py`,
  Signal/Score計算群)は本Phaseでも一切変更していない。
  `backtest/episode_analysis.py`への変更は新規関数の追加のみ(既存の
  `EpisodeMetrics`/`compute_episode_metrics`は無変更)。

---

## 15. Tests

pytest: 558 passed, 2 deselected
ruff: All checks passed
mypy: Success (96 source files)

新規テストファイル: `test_backtest_day_cluster_bootstrap.py`,
`test_backtest_block_bootstrap.py`, `test_backtest_event_concentration.py`,
`test_backtest_timing_shift.py`, `test_backtest_episode_analysis.py`
(拡張), `test_pipeline_run_phase9_analysis.py`, `test_phase9_no_lookahead.py`,
`test_config.py`(拡張)。

---

## 結論

Phase 9は「long_oversold_reboundが2024年8月急落・急反発という特殊
イベントに依存した偶然なのか、それとも類似する急落局面に一般化可能な
regime-dependent edgeなのか」という問いに対し、**単純な二択ではない、
両方の側面を持つ結論**に到達した。

BEAR regime限定で見ると、優位性はたった1営業日で全損益の64.8%を占める
という極端な集中を示し、Timing Placebo Sweepでも実際のSignalタイミング
から数営業日離れるだけで優位性は急速に崩壊する。この意味で
**Primary: EVENT_DEPENDENT**である。

しかし、その支配的なイベントを完全に除外しても、BEAR regime限定で
High cost tierでもPF=10超を維持し、Day Cluster/Block Bootstrapという
より保守的な統計手法でもBEAR PFの信頼区間下限は1を明確に上回ったまま
であり、Combined(全regime込み)の結果は6つの事前固定ストレスシナリオ
全てでPF>1を維持した。この意味で**Secondary caveat: REGIME_DEPENDENT**
も同時に成立する。

結果が良くても悪くても、そのまま報告するという方針を貫いた。Signal・
Score・Backtest・WFO・Cost・Decision Frameworkは本Phase全体を通じて
一切変更しておらず、`long_oversold_rebound`の条件を有利に調整する対応も
行っていない。

**実運用への自動採用は行わない。**

**Phase 9完了。Phase 10以降には進まない。** Signal変更・追加・Parameter
optimization・実運用ロジック・自動売買・Streamlit UI・Paper Trading/
Forward Testのいずれにも進まない。

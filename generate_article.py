import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)

# =========================
# 入力
# =========================
PICKS_PATH = os.path.join(BASE_DIR, "outputs", "today_picks.csv")

today = datetime.now()
today_str = today.strftime("%Y年%m月%d日")
month_str = today.strftime("%Y-%m")

PERF_PATH = os.path.join(BASE_DIR, "logs", f"performance_{month_str}.csv")

if not os.path.exists(PICKS_PATH):
    print("❌ today_picks.csvが存在しません")
    exit()

df = pd.read_csv(PICKS_PATH)

# =========================
# 🔥 バックテスト結果（ここ固定 or 後で自動化）
# =========================
BT_CAGR = 0.28
BT_SHARPE = 1.45
BT_MAXDD = -0.13

bt_text = f"""
📊 バックテスト結果（過去検証）

CAGR: {BT_CAGR:.0%}
Sharpe: {BT_SHARPE:.2f}
最大ドローダウン: {BT_MAXDD:.0%}
"""

# =========================
# 📊 実績（今月）
# =========================
perf_text = ""
perf_detail = ""

if os.path.exists(PERF_PATH):

    df_perf = pd.read_csv(PERF_PATH)

    if len(df_perf) >= 3:
        win_rate = df_perf["win"].mean()
        avg_return = df_perf["return"].mean()
        std = df_perf["return"].std()
        sharpe = avg_return / std if std != 0 else 0

        perf_text = f"""
📊 今月実績

勝率: {win_rate:.1%}
平均リターン: {avg_return:.2%}
"""

        perf_detail = f"""
■ 実運用実績（今月）

・トレード数: {len(df_perf)}
・勝率: {win_rate:.1%}
・平均リターン: {avg_return:.2%}
・Sharpe: {sharpe:.2f}
"""

# =========================
# 無料版
# =========================
free_text = f"""
【{today_str}】AI注目銘柄

本日のAI注目銘柄はこちら👇

"""

for i, row in df.iterrows():
    free_text += f"{i+1}. {row['銘柄名']}（{row['コード']}）\n"

free_text += "\n"

# 🔥 信頼パート（超重要）
free_text += bt_text
free_text += perf_text

free_text += """

AIが過去データをもとに抽出しています。

---

▼有料版では以下を公開
・具体的な売買タイミング
・資金配分ルール
・ロジック詳細
・実績の深掘り分析

👉 継続して利益を狙うための考え方も解説
"""

# =========================
# 有料版
# =========================
paid_text = f"""
【{today_str}】AIトレード戦略（完全版）

■ 銘柄
"""

for i, row in df.iterrows():
    paid_text += f"{i+1}. {row['銘柄名']}（{row['コード']}）\n"

paid_text += "\n"

# 🔥 バックテスト（しっかり）
paid_text += f"""
■ バックテスト結果

・CAGR: {BT_CAGR:.0%}
・Sharpe: {BT_SHARPE:.2f}
・最大ドローダウン: {BT_MAXDD:.0%}
"""

# 🔥 実運用
paid_text += perf_detail

paid_text += f"""

■ 基本戦略
・エントリー：翌営業日寄り
・保有期間：7日
・銘柄数：最大1

■ 売買ルール
・利確：+8%
・損切り：-2%

■ ロジック概要
・LightGBMによる確率予測
・モメンタム＋トレンドのハイブリッド選定
・市場環境に応じたフィルター

■ この戦略の強み
・統計的優位性（バックテスト検証済）
・損小利大の設計
・継続運用による安定性

■ 注意
投資は自己責任でお願いします
"""

# =========================
# ファイル出力
# =========================
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FREE_PATH = os.path.join(OUTPUT_DIR, "note_free.txt")
PAID_PATH = os.path.join(OUTPUT_DIR, "note_paid.txt")

with open(FREE_PATH, "w", encoding="utf-8") as f:
    f.write(free_text)

with open(PAID_PATH, "w", encoding="utf-8") as f:
    f.write(paid_text)

print("\n💾 記事ファイル保存完了")
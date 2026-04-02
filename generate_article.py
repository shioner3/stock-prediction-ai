import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
INPUT_PATH = os.path.join(BASE_DIR, "outputs", "today_picks.csv")

if not os.path.exists(INPUT_PATH):
    print("❌ today_picks.csvが存在しません")
    exit()

df = pd.read_csv(INPUT_PATH)

today_str = datetime.now().strftime("%Y年%m月%d日")

# =========================
# 無料版
# =========================
free_text = f"""
【{today_str}】AI注目銘柄

本日のAI注目銘柄はこちら👇

"""

for i, row in df.iterrows():
    free_text += f"{i+1}. {row['銘柄名']}（{row['コード']}）\n"

free_text += """

※AIによるデータ分析に基づく抽出です

---

▼有料版では以下を公開
・選定ロジック
・売買ルール
・期待値の考え方
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
・市場状況に応じたフィルター

■ 注意
投資は自己責任でお願いします
"""

# =========================
# 出力
# =========================
print("\n===== 無料版 =====")
print(free_text)

print("\n===== 有料版 =====")
print(paid_text)
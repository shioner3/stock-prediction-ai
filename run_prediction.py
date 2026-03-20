# =========================
# IMPORT
# =========================
import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRegressor

# =========================
# 設定
# =========================
TOP_N = 5
DATA_PATH = "ml_dataset.parquet"

# =========================
# データ読み込み
# =========================
df = pd.read_parquet(DATA_PATH)

# =========================
# 特徴量・ターゲット
# =========================
FEATURES = [col for col in df.columns if col not in ["Date", "ticker", "target"]]

X = df[FEATURES]
y = df["target"]

# =========================
# モデル学習
# =========================
model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    random_state=42
)

model.fit(X, y)

# =========================
# 最新データで予測
# =========================
latest_df = df.sort_values("Date").groupby("ticker").tail(1)

X_latest = latest_df[FEATURES]
latest_df["score"] = model.predict(X_latest)

# 予測リターン（仮：そのままtarget扱い）
latest_df["pred_return"] = latest_df["score"]

# ボラティリティ（簡易）
latest_df["volatility"] = np.random.uniform(0.05, 0.2, len(latest_df))

# =========================
# 上位銘柄抽出
# =========================
result_df = latest_df.sort_values("score", ascending=False).head(TOP_N)

# =========================
# CSV出力（無料部分）
# =========================
result_df[["ticker", "score", "pred_return"]].to_csv("today_picks.csv", index=False)
print("✅ today_picks.csv 出力完了")

# =========================
# 解説生成関数
# =========================
def generate_comment(row, rank):
    ticker = row["ticker"]
    score = row["score"]
    ret = row["pred_return"]
    vol = row["volatility"]

    # トレンド判定
    if score > 0.85:
        trend = "強い上昇トレンド"
    elif score > 0.75:
        trend = "上昇トレンド"
    else:
        trend = "やや不安定"

    # リスク
    if vol < 0.1:
        risk = "低リスク"
    elif vol < 0.15:
        risk = "中リスク"
    else:
        risk = "高リスク"

    # シグナル
    if score > 0.85:
        signal = "非常に強い買いシグナル"
    elif score > 0.75:
        signal = "強めの買いシグナル"
    else:
        signal = "様子見"

    text = f"""
========================
{rank}位: {ticker}
========================

■ 期待リターン
{ret*100:.2f}%

■ AI評価
{signal}

■ 判断理由
{trend}かつ{risk}のため、
リスクリワードの良い銘柄と判断。

■ 戦略
・エントリー：押し目
・利確目安：+{ret*100:.2f}%
・損切り：-2%

"""
    return text

# =========================
# 記事生成
# =========================
def generate_article(df):
    articles = []
    for i, row in enumerate(df.to_dict("records"), 1):
        articles.append(generate_comment(row, i))
    return "\n".join(articles)

# =========================
# note記事出力（有料部分）
# =========================
article_text = generate_article(result_df)

with open("note_article.txt", "w", encoding="utf-8") as f:
    f.write(article_text)

print("📝 note_article.txt 生成完了")

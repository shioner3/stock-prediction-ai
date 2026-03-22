import pandas as pd
import numpy as np
import os
import pickle
from lightgbm import LGBMRegressor
from datetime import datetime
from pandas.tseries.offsets import BDay


# =========================
# 設定
# =========================
TOP_N = 5

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


FEATURES = [
    "Return_1_rank",
    "MA5_ratio_rank",
    "MA25_ratio_rank",
    "MA75_ratio_rank",
    "Volatility_rank",
    "Volume_change_rank",
    "HL_range_rank",
    "RSI_rank"
]

TARGET = "Target"


# =========================
# 🔥 モデル準拠 戦略
# =========================
def generate_model_strategy(row):
    rank = row["PredRank"]

    if rank == 1:
        confidence = "最高"
    elif rank <= 3:
        confidence = "高"
    else:
        confidence = "中"

    return f"""
■ AI評価
{rank}位（信頼度: {confidence}）

■ エントリー
当日の寄付き or 引けで即エントリー

■ 保有期間
5営業日

■ 利確
5営業日後に決済

■ 損切り
-3%

■ 根拠
本モデルは5営業日後のリターンを予測しているため、
即エントリーが最も再現性が高い。
"""


# =========================
# 🔥 日次判断AI
# =========================
def generate_daily_decision(picks_df, full_df):

    market_score = full_df["Pred"].mean()

    if market_score > 0.55:
        trend = "強気"
        action = "積極エントリー"
        position = "80〜100%"
        max_n = 5

    elif market_score > 0.52:
        trend = "やや強気"
        action = "選別エントリー"
        position = "50〜70%"
        max_n = 3

    elif market_score > 0.5:
        trend = "中立"
        action = "様子見"
        position = "30%"
        max_n = 1

    else:
        trend = "弱気"
        action = "見送り"
        position = "0%"
        max_n = 0

    text = f"""
========================
■ 本日のAI判断
========================

市場評価: {trend}
行動: {action}
推奨ポジション: {position}

"""

    if max_n == 0:
        text += "👉 本日はノートレード推奨\n"
    else:
        selected = picks_df.head(max_n)
        text += "\n■ 採用銘柄\n"

        for i, row in enumerate(selected.itertuples(), 1):
            text += f"{i}. {row.銘柄名} ({row.コード})\n"

    return text


# =========================
# 🔥 記事生成
# =========================
def generate_article(df, daily_comment):
    texts = [daily_comment]

    for _, row in df.iterrows():
        text = f"""
========================
{row["PredRank"]}位: {row["銘柄名"]} ({row["コード"]})
========================

{row["Strategy"]}
"""
        texts.append(text)

    return "\n".join(texts)


# =========================
# データ読み込み
# =========================
print("Loading dataset...")

df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

latest_date = df["Date"].max()
print("Prediction Date:", latest_date.date())


# =========================
# モデル
# =========================
retrain = (latest_date.weekday() == 0) or (not os.path.exists(MODEL_PATH))

if retrain:
    print("Weekly retrain...")
    train = df[df["Date"] < latest_date]

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(train[FEATURES], train[TARGET])

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

else:
    print("Loading existing model...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)


# =========================
# 今日データ
# =========================
today = df[df["Date"] == latest_date].copy()

today["Pred"] = model.predict(today[FEATURES])
today["Rank"] = today["Pred"].rank(ascending=False)


# =========================
# 銘柄名
# =========================
df_info = pd.read_csv(os.path.join(BASE_DIR, "data_j.csv"), dtype=str)

ticker_to_name = dict(zip(df_info["コード"], df_info["銘柄名"]))


# =========================
# picks
# =========================
picks = today.nsmallest(TOP_N, "Rank").copy()

picks["コード"] = picks["Ticker"].str.replace(".T", "", regex=False)
picks["銘柄名"] = picks["コード"].map(ticker_to_name)
picks["PredRank"] = range(1, len(picks) + 1)

picks = picks[["Ticker", "コード", "銘柄名", "Pred", "PredRank"]]


# =========================
# 🔥 戦略追加
# =========================
picks["Strategy"] = picks.apply(generate_model_strategy, axis=1)


# =========================
# 🔥 日次判断
# =========================
daily_comment = generate_daily_decision(picks, today)


# =========================
# CSV（無料）
# =========================
picks[["Ticker", "コード", "銘柄名"]].to_csv("today_picks.csv", index=False)


# =========================
# 🔥 記事生成（有料）
# =========================
article = generate_article(picks, daily_comment)

with open("note_article.txt", "w", encoding="utf-8") as f:
    f.write(article)

print("📝 note_article.txt 生成完了")


# =========================
# 🔥 実績ログ
# =========================
today_dt = datetime.now()
target_dt = today_dt + BDay(5)

picks["predict_date"] = today_dt.strftime("%Y-%m-%d")
picks["target_date"] = target_dt.strftime("%Y-%m-%d")

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

picks.to_csv(
    os.path.join(LOG_DIR, "predictions.csv"),
    mode="a",
    header=not os.path.exists(os.path.join(LOG_DIR, "predictions.csv")),
    index=False
)

print("✅ 完全処理完了")
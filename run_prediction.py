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
TOP_PREMIUM = 20   # ←追加（有料20銘柄）

BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest.parquet")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

FREE_CSV_PATH = "today_picks_free.csv"
PREMIUM_CSV_PATH = os.path.join(LOG_DIR, "today_picks_premium.csv")

ARTICLE_PATH = "note_article.txt"

month_str = datetime.now().strftime("%Y-%m")
PRED_LOG_PATH = os.path.join(LOG_DIR, f"predictions_{month_str}.csv")

# 完全性判定
MIN_COUNT = 3000

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
# ユーティリティ
# =========================
def normalize_columns(df):
    rename_map = {}
    if "コード" not in df.columns and "Ticker" in df.columns:
        rename_map["Ticker"] = "コード"
    if "銘柄名" not in df.columns and "Name" in df.columns:
        rename_map["Name"] = "銘柄名"
    return df.rename(columns=rename_map) if rename_map else df

def normalize(df):
    df = df.copy()
    df["PredRank"] = df["Pred"].rank(ascending=False, method="first")
    df = df.sort_values("PredRank")
    df["PredRank"] = range(1, len(df) + 1)
    return df

def get_regime(score):
    if score > 0.55:
        return "strong"
    elif score > 0.52:
        return "slightly_strong"
    elif score > 0.5:
        return "neutral"
    else:
        return "weak"

# =========================
# データ読み込み
# =========================
train_df = pd.read_parquet(TRAIN_DATA_PATH)
predict_df = pd.read_parquet(PREDICT_DATA_PATH)

train_df = normalize_columns(train_df)
predict_df = normalize_columns(predict_df)

train_df["Date"] = pd.to_datetime(train_df["Date"])
predict_df["Date"] = pd.to_datetime(predict_df["Date"])

# =========================
# 🔥 有効な最新日を取得（超重要）
# =========================
counts = predict_df["Date"].value_counts()

print("\n=== 最新付近チェック ===")
print(counts.sort_index().tail(10))

valid_dates = counts[counts >= MIN_COUNT].index

if len(valid_dates) == 0:
    raise ValueError("有効な日がありません")

latest_date = max(valid_dates)

print("\n=== 使用日 ===")
print("latest_valid:", latest_date)
print("銘柄数:", counts[latest_date])

# =========================
# モデル
# =========================
if not os.path.exists(MODEL_PATH):
    model = LGBMRegressor()
    model.fit(train_df[FEATURES], train_df[TARGET])
    pickle.dump(model, open(MODEL_PATH, "wb"))
else:
    model = pickle.load(open(MODEL_PATH, "rb"))

# =========================
# 予測
# =========================
today = predict_df[predict_df["Date"] == latest_date].copy()

# 念のためNaN除去
today = today.dropna(subset=FEATURES)

today["Pred"] = model.predict(today[FEATURES])
today = normalize(today)

print("\n=== 予測件数 ===")
print(len(today))

# =========================
# CSV出力
# =========================

# 無料（TOP5）
today.head(TOP_N)[["コード", "銘柄名", "PredRank"]]\
    .rename(columns={"PredRank": "順位"})\
    .to_csv(FREE_CSV_PATH, index=False)

# 有料（TOP20）
premium_df = today.head(TOP_PREMIUM).copy()

premium_df["regime"] = get_regime(today["Pred"].mean())
premium_df["predict_date"] = datetime.now().strftime("%Y-%m-%d")
premium_df["target_date"] = (datetime.now() + BDay(5)).strftime("%Y-%m-%d")

premium_df.to_csv(PREMIUM_CSV_PATH, index=False)

# =========================
# 記事生成（簡略）
# =========================
text = "■ 上位20銘柄\n\n"

for _, row in premium_df.iterrows():
    text += f"{int(row['PredRank'])}位 {row['銘柄名']} ({row['コード']})\n"

with open(ARTICLE_PATH, "w", encoding="utf-8") as f:
    f.write(text)

# =========================
# LOG
# =========================
log_df = premium_df[
    ["コード", "銘柄名", "Pred", "PredRank", "regime", "predict_date", "target_date"]
]

if os.path.exists(PRED_LOG_PATH):
    old = pd.read_csv(PRED_LOG_PATH)
    log_df = pd.concat([old, log_df], ignore_index=True)

log_df.to_csv(PRED_LOG_PATH, index=False)

print("\n✅ 完了（20銘柄出力・安定版）")
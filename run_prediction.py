import pandas as pd
import numpy as np
import os
import pickle
from lightgbm import LGBMRegressor
from datetime import datetime

# =========================
# 設定
# =========================
TOP_N = 5
HOLD_DAYS = 3

BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest.parquet")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

ARTICLE_PATH = "note_article.txt"

month_str = datetime.now().strftime("%Y-%m")
PRED_LOG_PATH = os.path.join(LOG_DIR, f"predictions_{month_str}.csv")

PERF_LOG_PATH = os.path.join(LOG_DIR, "performance.csv")

# =========================
# 特徴量
# =========================
FEATURES = [
    "Return_1_rank",
    "Return_3_rank",
    "MA3_ratio_rank",
    "MA5_ratio_rank",
    "MA10_ratio_rank",
    "Volatility_rank",
    "Volume_change_rank",
    "Volume_ratio_rank",
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
    if score > 0.56:
        return "strong"
    elif score > 0.53:
        return "slightly_strong"
    elif score > 0.51:
        return "neutral"
    else:
        return "weak"


def load_performance():
    if not os.path.exists(PERF_LOG_PATH):
        return None

    df = pd.read_csv(PERF_LOG_PATH)
    if len(df) < 10:
        return None

    df = df.tail(100)

    return {
        "win_rate": df["win"].mean(),
        "avg_return": df["return"].mean(),
        "sharpe": df["return"].mean() / df["return"].std() if df["return"].std() != 0 else 0
    }


# =========================
# モデル
# =========================
train_df = pd.read_parquet(TRAIN_DATA_PATH)
predict_df = pd.read_parquet(PREDICT_DATA_PATH)

train_df = normalize_columns(train_df)
predict_df = normalize_columns(predict_df)

train_df["Date"] = pd.to_datetime(train_df["Date"])
predict_df["Date"] = pd.to_datetime(predict_df["Date"])

latest_date = predict_df["Date"].max()

# 毎回再学習
model = LGBMRegressor(n_estimators=200, learning_rate=0.05)
model.fit(train_df[FEATURES], train_df[TARGET])

pickle.dump(model, open(MODEL_PATH, "wb"))

# =========================
# 予測
# =========================
today = predict_df[predict_df["Date"] == latest_date].copy()

today["Pred"] = model.predict(today[FEATURES])

# =========================
# 🔥 地雷フィルター（超重要）
# =========================

# 決算（事前に列がある前提）
if "is_earnings" in today.columns:
    today = today[today["is_earnings"] == 0]

# ストップ高翌日
if "limit_up_flag" in today.columns:
    today = today[today["limit_up_flag"] == 0]

# 出来高（流動性）
if "Volume" in today.columns:
    today = today[today["Volume"] > 100000]

# 出来高異常
if "Volume_ratio" in today.columns:
    today = today[today["Volume_ratio"] < 3]

# =========================
# スコアフィルター
# =========================
today = today[today["Pred"] > 0.52]

if len(today) == 0:
    print("⚠️ フィルター後に銘柄なし")
    exit()

# =========================
# ランキング
# =========================
today = normalize(today)

# =========================
# レジーム
# =========================
market_score = today["Pred"].mean()
regime = get_regime(market_score)

# =========================
# ログ保存（重要）
# =========================
today["predict_date"] = latest_date
today["target_date"] = latest_date + pd.Timedelta(days=HOLD_DAYS)

if os.path.exists(PRED_LOG_PATH):
    today.to_csv(PRED_LOG_PATH, mode="a", header=False, index=False)
else:
    today.to_csv(PRED_LOG_PATH, index=False)

# =========================
# 出力
# =========================
print("\n=== 今日の銘柄 ===")
print(today.head(TOP_N)[["コード", "銘柄名", "Pred"]])

print("\n市場状態:", regime)
print("銘柄数:", len(today))
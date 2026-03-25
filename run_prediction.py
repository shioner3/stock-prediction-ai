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
WEAK_TOP_PERCENT = 0.01

BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest.parquet")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

FREE_CSV_PATH = "today_picks_free.csv"
PREMIUM_CSV_PATH = os.path.join(LOG_DIR, "today_picks_premium.csv")

month_str = datetime.now().strftime("%Y-%m")
PRED_LOG_PATH = os.path.join(LOG_DIR, f"predictions_{month_str}.csv")

PERF_LOG_PATH = os.path.join(LOG_DIR, "performance.csv")


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
# 列名吸収
# =========================
def normalize_columns(df):
    rename_map = {}

    if "コード" not in df.columns and "Ticker" in df.columns:
        rename_map["Ticker"] = "コード"

    if "銘柄名" not in df.columns and "Name" in df.columns:
        rename_map["Name"] = "銘柄名"

    return df.rename(columns=rename_map) if rename_map else df


# =========================
# ランク
# =========================
def normalize(df):
    df = df.copy()
    df["PredRank"] = df["Pred"].rank(ascending=False, method="first")
    df = df.sort_values("PredRank")
    df["PredRank"] = range(1, len(df) + 1)
    return df


# =========================
# レジーム
# =========================
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
# 実績読み込み
# =========================
def load_performance():
    if not os.path.exists(PERF_LOG_PATH):
        return None

    df = pd.read_csv(PERF_LOG_PATH)
    if len(df) < 10:
        return None

    df = df.tail(100)

    result = {
        "all": {
            "win_rate": df["win"].mean(),
            "avg_return": df["return"].mean(),
            "sharpe": df["return"].mean() / df["return"].std() if df["return"].std() != 0 else 0
        },
        "regime": {}
    }

    for r in ["strong", "slightly_strong", "neutral", "weak"]:
        df_r = df[df["regime"] == r]
        if len(df_r) < 5:
            continue

        result["regime"][r] = {
            "win_rate": df_r["win"].mean(),
            "avg_return": df_r["return"].mean(),
            "sharpe": df_r["return"].mean() / df_r["return"].std() if df_r["return"].std() != 0 else 0
        }

    return result


# =========================
# データ読み込み（分離）
# =========================
train_df = pd.read_parquet(TRAIN_DATA_PATH)
predict_df = pd.read_parquet(PREDICT_DATA_PATH)

train_df = normalize_columns(train_df)
predict_df = normalize_columns(predict_df)

train_df["Date"] = pd.to_datetime(train_df["Date"])
predict_df["Date"] = pd.to_datetime(predict_df["Date"])

latest_date = predict_df["Date"].max()

print("\n=== PREDICTION DEBUG ===")
print("train latest:", train_df["Date"].max())
print("predict latest:", latest_date)
print("today rows:", len(predict_df[predict_df["Date"] == latest_date]))
print("========================")


# =========================
# モデル
# =========================
retrain = (latest_date.weekday() == 0) or (not os.path.exists(MODEL_PATH))

if retrain:
    train = train_df.copy()
    model = LGBMRegressor()
    model.fit(train[FEATURES], train[TARGET])

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
else:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)


# =========================
# 予測（最新データ）
# =========================
today = predict_df[predict_df["Date"] == latest_date].copy()

today["Pred"] = model.predict(today[FEATURES])
today = normalize(today)


# =========================
# レジーム
# =========================
market_score = today["Pred"].mean()
regime = get_regime(market_score)

# =========================
# FREE CSV
# =========================
free_csv = today.head(TOP_N)[["コード", "銘柄名", "PredRank"]].copy()
free_csv = free_csv.rename(columns={"PredRank": "順位"})
free_csv.to_csv(FREE_CSV_PATH, index=False)


# =========================
# PREMIUM CSV
# =========================
premium_df = today.copy()
premium_df["regime"] = regime
premium_df["predict_date"] = datetime.now().strftime("%Y-%m-%d")
premium_df["target_date"] = (datetime.now() + BDay(5)).strftime("%Y-%m-%d")

premium_df.to_csv(PREMIUM_CSV_PATH, index=False)


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


print("✅ 完了（学習/予測分離版）")
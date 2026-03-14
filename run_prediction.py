import pandas as pd
import numpy as np
import os
import pickle
from lightgbm import LGBMRegressor


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
# データ読み込み
# =========================
print("Loading dataset...")

df = pd.read_parquet(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])


# =========================
# 最新日取得
# =========================
latest_date = df["Date"].max()

print("Prediction Date:", latest_date.date())


# =========================
# モデル読み込み or 学習
# =========================

if os.path.exists(MODEL_PATH):

    print("Loading model...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

else:

    print("Training model...")

    train = df[df["Date"] < latest_date]

    X_train = train[FEATURES]
    y_train = train[TARGET]

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


# =========================
# 今日のデータ
# =========================

today = df[df["Date"] == latest_date].copy()

X_today = today[FEATURES]


# =========================
# 予測
# =========================

today["Pred"] = model.predict(X_today)

# 順位化
today["Rank"] = today["Pred"].rank(ascending=False)


# =========================
# 上位銘柄抽出
# =========================

picks = today.nsmallest(TOP_N, "Rank")

picks = picks[["Ticker", "Rank"]]


# =========================
# 出力
# =========================

print()
print("===== Today Picks =====")
print(picks)

# CSV保存
OUTPUT_PATH = os.path.join(BASE_DIR, "today_picks.csv")
picks.to_csv(OUTPUT_PATH, index=False)

print()
print("Saved:", OUTPUT_PATH)

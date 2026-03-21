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
# 再学習判定
# =========================
weekday = latest_date.weekday()  # 月=0

retrain = False

if weekday == 0:  # 月曜日
    retrain = True

if not os.path.exists(MODEL_PATH):
    retrain = True


# =========================
# モデル学習
# =========================
if retrain:

    print("Weekly retrain...")

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

else:

    print("Loading existing model...")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)


# =========================
# 今日のデータ
# =========================
today = df[df["Date"] == latest_date].copy()

X_today = today[FEATURES]


# =========================
# 予測
# =========================
today["Pred"] = model.predict(X_today)

today["Rank"] = today["Pred"].rank(ascending=False)


# =========================
# 銘柄名読み込み
# =========================
DATA_J_PATH = os.path.join(BASE_DIR, "data_j.csv")
df_info = pd.read_csv(DATA_J_PATH, dtype=str)

ticker_to_name = dict(
    zip(
        df_info["コード"].str.strip(),
        df_info["銘柄名"].str.strip()
    )
)


# =========================
# 上位銘柄
# =========================
picks = today.nsmallest(TOP_N, "Rank").copy()

picks["コード"] = picks["Ticker"].str.replace(".T", "", regex=False)
picks["銘柄名"] = picks["コード"].map(ticker_to_name)

picks["PredRank"] = range(1, len(picks) + 1)

picks = picks[["コード", "銘柄名", "PredRank"]]


# =========================
# 出力
# =========================
print()
print("===== Today Picks =====")
print(picks)

OUTPUT_PATH = os.path.join(BASE_DIR, "today_picks.csv")

picks.to_csv(OUTPUT_PATH, index=False)

print()
print("Saved:", OUTPUT_PATH)s

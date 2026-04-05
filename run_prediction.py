import pandas as pd
import numpy as np
import os
import pickle
from lightgbm import LGBMRegressor
from datetime import datetime

# =========================
# 設定
# =========================
TOP_N = 3
HOLD_DAYS = 3

BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest.parquet")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# =========================
# データ
# =========================
train_df = pd.read_parquet(TRAIN_DATA_PATH)
predict_df = pd.read_parquet(PREDICT_DATA_PATH)

FEATURES = [c for c in train_df.columns if c not in ["Date","Ticker","Name","Target"]]

# =========================
# モデル
# =========================
model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    random_state=42
)

model.fit(train_df[FEATURES], train_df["Target"])

pickle.dump(model, open(MODEL_PATH, "wb"))

# =========================
# 予測
# =========================
today = predict_df.copy()
today["Pred"] = model.predict(today[FEATURES])

# =========================
# ランキング
# =========================
today = today.sort_values("Pred", ascending=False).head(TOP_N)

today["PredRank"] = range(1, len(today)+1)

print("\n=== 今日の銘柄 ===")
print(today[["Ticker","Name","Pred","PredRank"]])
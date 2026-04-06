import pandas as pd
import numpy as np
import os
import pickle
from lightgbm import LGBMRegressor

# =========================
# 設定
# =========================
TOP_N = 3
HOLD_DAYS = 7

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

# 保存
pickle.dump(model, open(MODEL_PATH, "wb"))

# =========================
# 予測
# =========================
today = predict_df.copy()

# 🔥 生値スコア
today["raw_score"] = model.predict(today[FEATURES])

# 🔥 rank化（これが本質）
today["score"] = today["raw_score"].rank(pct=True)

# =========================
# 上位選択（rankベース）
# =========================
today = today.sort_values("score", ascending=False).head(TOP_N)

today["PredRank"] = range(1, len(today)+1)

# =========================
# 出力
# =========================
print("\n=== 今日の銘柄（rankベース） ===")
print(today[["Ticker","Name","raw_score","score","PredRank"]])
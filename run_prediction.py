import pandas as pd
import numpy as np
import os
import pickle
from lightgbm import LGBMRanker

# =========================
# 設定
# =========================
TOP_N = 5
N_CLASS = 30

BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest.parquet")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# =========================
# データ
# =========================
train_df = pd.read_parquet(TRAIN_DATA_PATH)
predict_df = pd.read_parquet(PREDICT_DATA_PATH)

# =========================
# FEATURES
# =========================
FEATURES = [
    "Return_1","Return_3",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio",
    "HL_range",
    "Rel_Return_1",
    "Trend_5_z","Trend_10_z","Trend_diff",
    "DD_5","DD_10",
    "TrendVol",
    "Volume_Z",
    "Gap",
    "Volatility_change",
    "Momentum_acc",
    "Return_1_rank","Return_3_rank",
    "Volume_ratio_rank","Trend_5_z_rank","HL_range_rank",
    "Market_Z","Market_Trend",
    "DayOfWeek"
]

# =========================
# 前処理
# =========================
train_df = train_df.dropna(subset=FEATURES + ["Target"])
predict_df = predict_df.dropna(subset=FEATURES)

# =========================
# TargetClass
# =========================
def make_target_class(x):
    try:
        return pd.qcut(x, q=N_CLASS, labels=False, duplicates="drop")
    except:
        return pd.cut(x, bins=min(N_CLASS, len(x)), labels=False)

train_df["TargetClass"] = train_df.groupby("Date")["Target"].transform(make_target_class)
train_df = train_df.dropna(subset=["TargetClass"])
train_df["TargetClass"] = train_df["TargetClass"].astype(int)

group = train_df.groupby("Date").size().to_list()

# =========================
# モデル
# =========================
model = LGBMRanker(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(train_df[FEATURES], train_df["TargetClass"], group=group)

# =========================
# 予測
# =========================
today = predict_df.copy()

today["raw_score"] = model.predict(today[FEATURES])

# 🔥 rankが最強
today["score"] = today.groupby("Date")["raw_score"].rank(pct=True)

# =========================
# 上位抽出
# =========================
today = today.sort_values("score", ascending=False).head(TOP_N)
today["PredRank"] = range(1, len(today)+1)

# =========================
# weight（均等でOK）
# =========================
today["weight"] = 1 / len(today)

# =========================
# 出力
# =========================
print(today[[
    "Ticker","Name",
    "score","PredRank","weight"
]])
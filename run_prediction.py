import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRanker

# =========================
# 設定
# =========================
TOP_N = 3

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
# FEATURES（featureと完全一致）
# =========================
FEATURES = [
    "Return_1","Return_3",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio",
    "HL_range",
    "Rel_Return_1",
    "Trend_5_z","Trend_10_z","Trend_diff",
    "Gap","Volatility_change","Momentum_acc",
    "DD_5","DD_10",
    "TrendVol","Volume_Z",
    "Return_1_rank","Volume_ratio_rank",
    "Trend_5_z_rank","TrendVol_rank",
    "Market_Z","Market_Trend"
]

# =========================
# 前処理
# =========================
train_df = train_df.dropna(subset=FEATURES + ["Target"])
predict_df = predict_df.dropna(subset=FEATURES)

# =========================
# Ranker（シンプル化）
# =========================
group = train_df.groupby("Date").size().values.tolist()

model = LGBMRanker(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(train_df[FEATURES], train_df["Target"], group=group)

# =========================
# 予測
# =========================
today = predict_df.copy()

today["score_raw"] = model.predict(today[FEATURES])

# 日内ランキング（重要）
today["score"] = today.groupby("Date")["score_raw"].rank(pct=True)

# =========================
# TOP選択（ここだけが戦略）
# =========================
today = today.sort_values("score", ascending=False).head(TOP_N)
today["rank"] = range(1, len(today) + 1)

# =========================
# weight（シンプルで安定）
# =========================
today["weight"] = today["score"] / today["score"].sum()

# =========================
# 出力
# =========================
print("\n=== 今日の銘柄 ===")

print(today[[
    "Ticker",
    "score",
    "Trend_5_z",
    "TrendVol",
    "DD_5",
    "weight",
    "rank"
]])

print("\n件数:", len(today))
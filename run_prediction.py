import pandas as pd
import numpy as np
import os
import pickle
from lightgbm import LGBMRanker

# =========================
# 設定
# =========================
TOP_N = 3
TOP_RATE = 0.1
HOLD_DAYS = 7

USE_MARKET_FILTER = True

BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest.parquet")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# =========================
# データ読み込み
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
    "Gap",
    "Volatility_change",
    "Momentum_acc",
    "Return_1_rank","Return_3_rank",
    "Volume_ratio_rank","Trend_5_z_rank","HL_range_rank",
    "Market_Z","Market_Trend",
    "DayOfWeek"
]

# =========================
# 学習データ前処理（🔥ここ重要）
# =========================
train_df = train_df.sort_values("Date").copy()

# 🔥 Rankラベル生成
train_df["TargetRank"] = train_df.groupby("Date")["Target"].rank(method="first")

# group作成
group = train_df.groupby("Date").size().to_list()

# =========================
# モデル
# =========================
model = LGBMRanker(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# 🔥 Target → TargetRankに変更
model.fit(
    train_df[FEATURES],
    train_df["TargetRank"],
    group=group
)

# =========================
# モデル保存
# =========================
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

# =========================
# 予測
# =========================
today = predict_df.copy()

today["raw_score"] = model.predict(today[FEATURES])

# クロスセクションrank
today["score"] = today["raw_score"].rank(pct=True)

# =========================
# 市場フィルター
# =========================
if USE_MARKET_FILTER:
    today = today[today["Market_Trend"] > 0]

# =========================
# TOP_RATEフィルター
# =========================
today = today[today["score"] >= (1 - TOP_RATE)]

# =========================
# 最終選抜
# =========================
today = today.sort_values("score", ascending=False).head(TOP_N)

today["PredRank"] = range(1, len(today)+1)

# =========================
# 重み（score × trend）
# =========================
today["weight_raw"] = today["score"] * (1 + today["Trend_5_z"].clip(-1, 1))

# ゼロ除算防止
if today["weight_raw"].sum() > 0:
    today["weight"] = today["weight_raw"] / today["weight_raw"].sum()
else:
    today["weight"] = 1.0 / len(today)

# =========================
# 出力
# =========================
print("\n=== 今日の銘柄（Ranker最終版） ===")
print(today[[
    "Ticker","Name",
    "raw_score","score",
    "Trend_5_z",
    "weight",
    "PredRank"
]])

# =========================
# デバッグ
# =========================
print("\n=== SCORE分布 ===")
print(today["score"].describe())

print("\n=== 件数 ===")
print(len(today))

print("\n=== 使用特徴量数 ===")
print(len(FEATURES))
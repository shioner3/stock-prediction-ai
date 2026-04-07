import pandas as pd
import numpy as np
import os
import pickle
from lightgbm import LGBMRanker

# =========================
# 設定（重要パラメータ）
# =========================
TOP_N = 3
TOP_RATE = 0.1        # ← ここ調整ポイント（0.01〜0.1）
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
# FEATURES（完全一致）
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
# ランキング学習用 group 作成
# =========================
train_df = train_df.sort_values("Date")
group = train_df.groupby("Date").size().to_list()

# =========================
# モデル（🔥 Ranker化）
# =========================
model = LGBMRanker(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# 学習
model.fit(
    train_df[FEATURES],
    train_df["Target"],
    group=group
)

# 保存
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

# =========================
# 予測
# =========================
today = predict_df.copy()

# rawスコア
today["raw_score"] = model.predict(today[FEATURES])

# rank化（クロスセクション）
today["score"] = today["raw_score"].rank(pct=True)

# =========================
# 🔥 市場フィルター（DD削減）
# =========================
if USE_MARKET_FILTER:
    today = today[today["Market_Trend"] > 0]

# =========================
# 🔥 TOP_RATEフィルター
# =========================
today = today[today["score"] >= (1 - TOP_RATE)]

# =========================
# 🔥 最終選抜
# =========================
today = today.sort_values("score", ascending=False).head(TOP_N)

today["PredRank"] = range(1, len(today)+1)

# =========================
# 🔥 重み（最重要改善）
# =========================
# スコア × トレンド
today["weight_raw"] = today["score"] * (1 + today["Trend_5_z"].clip(-1, 1))

# 正規化
today["weight"] = today["weight_raw"] / today["weight_raw"].sum()

# =========================
# 出力
# =========================
print("\n=== 今日の銘柄（収益化版） ===")
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
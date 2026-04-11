import pandas as pd
import numpy as np
import os
import pickle
import lightgbm as lgb
from lightgbm import LGBMRanker

# =========================
# 設定
# =========================
TOP_N = 5
TOP_RATE = 0.014
HOLD_DAYS = 10  # ベース

USE_MARKET_FILTER = True
N_CLASS = 30

TRAIN_START_DATE = "2018-01-01"

BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest.parquet")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# =========================
# データ読み込み
# =========================
train_df = pd.read_parquet(TRAIN_DATA_PATH)
predict_df = pd.read_parquet(PREDICT_DATA_PATH)

train_df = train_df[train_df["Date"] >= TRAIN_START_DATE].copy()

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
# 欠損除去
# =========================
train_df = train_df.dropna(subset=FEATURES + ["Target"]).copy()
predict_df = predict_df.dropna(subset=FEATURES).copy()

# =========================
# Target
# =========================
train_df = train_df.sort_values("Date").copy()

def make_target_class(x):
    try:
        return pd.qcut(x, q=N_CLASS, labels=False, duplicates="drop")
    except:
        return pd.cut(x, bins=min(N_CLASS, len(x)), labels=False)

train_df["TargetClass"] = train_df.groupby("Date")["Target"].transform(make_target_class)
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
    random_state=42,
    n_jobs=-1
)

model.fit(
    train_df[FEATURES],
    train_df["TargetClass"],
    group=group,
    eval_set=[(train_df[FEATURES], train_df["TargetClass"])],
    eval_group=[group],
    eval_at=[5],
    callbacks=[lgb.early_stopping(50)]
)

# =========================
# 保存
# =========================
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)


# =========================
# 予測
# =========================
today = predict_df.copy()

today["raw_score"] = model.predict(today[FEATURES])

# 🔥 前日シグナルにする
today["raw_score_shift"] = today.groupby("Ticker")["raw_score"].shift(1)

# ランク化
today["score"] = today["raw_score_shift"].rank(pct=True)

# シフトでNaN消す
today = today.dropna(subset=["score"])

# =========================
# 市場フィルター
# =========================
if USE_MARKET_FILTER:
    today = today[today["Market_Trend"] > 0]

# =========================
# TOP_RATE
# =========================
today = today[today["score"] >= (1 - TOP_RATE)]

# =========================
# フィルタ
# =========================
today = today[today["TrendVol"] < today["TrendVol"].quantile(0.7)]
today = today[today["DD_5"] > -0.05]

# =========================
# 最終選抜
# =========================
today = today.sort_values("score", ascending=False).head(TOP_N)
today["PredRank"] = range(1, len(today)+1)


# =========================
# 重み
# =========================
today["weight_raw"] = today["score"]

today["weight"] = today["weight_raw"] / today["weight_raw"].sum()

# =========================
# 出力
# =========================
print("\n=== 今日の銘柄 ===")
print(today[[
    "Ticker","Name",
    "raw_score","score",
    "Trend_5_z","TrendVol","DD_5",
    "hold_days",
    "weight",
    "PredRank"
]])

# =========================
# デバッグ
# =========================
print("\n=== HOLD分布 ===")
print(today["hold_days"].describe())

print("\n=== SCORE分布 ===")
print(today["score"].describe())

print("\n=== 件数 ===")
print(len(today))
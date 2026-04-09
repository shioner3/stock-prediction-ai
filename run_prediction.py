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
TOP_RATE = 0.1
HOLD_DAYS = 10

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

# 古いデータ削減
train_df = train_df[train_df["Date"] >= TRAIN_START_DATE].copy()

# =========================
# FEATURES（🔥ここが最重要）
# =========================
FEATURES = [
    # ===== 基本 =====
    "Return_1","Return_3",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio",
    "HL_range",
    "Rel_Return_1",

    # ===== トレンド =====
    "Trend_5_z","Trend_10_z","Trend_diff",

    # ===== ドローダウン系（NEW）★★★★★
    "DD_5","DD_10",

    # ===== トレンド安定性（NEW）★★★★★
    "TrendVol",

    # ===== 出来高Z（NEW）★★★★☆
    "Volume_Z",

    # ===== 追加 =====
    "Gap",
    "Volatility_change",
    "Momentum_acc",

    # ===== クロスセクション =====
    "Return_1_rank","Return_3_rank",
    "Volume_ratio_rank","Trend_5_z_rank","HL_range_rank",

    # ===== 市場 =====
    "Market_Z","Market_Trend",

    # ===== 時間 =====
    "DayOfWeek"
]

# =========================
# 欠損チェック（安全化）
# =========================
train_df = train_df.dropna(subset=FEATURES + ["Target"]).copy()
predict_df = predict_df.dropna(subset=FEATURES).copy()

# =========================
# 学習データ前処理
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

# =========================
# 学習
# =========================
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
# 🔥 連敗対策（追加するとさらに良い）
# =========================
# 安定トレンド優先
today = today[today["TrendVol"] < today["TrendVol"].quantile(0.7)]

# ドローダウン回避
today = today[today["DD_5"] > -0.05]

# =========================
# 最終選抜
# =========================
today = today.sort_values("score", ascending=False).head(TOP_N)
today["PredRank"] = range(1, len(today)+1)

# =========================
# 重み（改良版）
# =========================
today["weight_raw"] = (
    today["score"]**2 *
    (1 + today["Trend_5_z"].clip(0, 2)) *
    (1 - today["TrendVol"].clip(0, 1))
)

if today["weight_raw"].sum() > 0:
    today["weight"] = today["weight_raw"] / today["weight_raw"].sum()
else:
    today["weight"] = 1.0 / len(today)

# =========================
# 出力
# =========================
print("\n=== 今日の銘柄（最終版） ===")
print(today[[
    "Ticker","Name",
    "raw_score","score",
    "Trend_5_z","TrendVol","DD_5",
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
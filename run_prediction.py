import pandas as pd
import numpy as np
import os
import pickle
from lightgbm import LGBMRegressor

# =========================
# 設定
# =========================
TOP_N = 3
HOLD_DAYS = 7   # ← feature側と統一（重要）

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
    # ベース
    "Return_1","Return_3",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio",
    "HL_range",
    "Rel_Return_1",

    # トレンド
    "Trend_5_z","Trend_10_z","Trend_diff",

    # 追加
    "Gap",
    "Volatility_change",
    "Momentum_acc",

    # クロスセクション🔥
    "Return_1_rank","Return_3_rank",
    "Volume_ratio_rank","Trend_5_z_rank","HL_range_rank",

    # 市場
    "Market_Z","Market_Trend",

    # 時間
    "DayOfWeek"
]

# =========================
# モデル
# =========================
model = LGBMRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# 学習
model.fit(train_df[FEATURES], train_df["Target"])

# 保存
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

# =========================
# 予測
# =========================
today = predict_df.copy()

# 生スコア
today["raw_score"] = model.predict(today[FEATURES])

# rank化（クロスセクション）
today["score"] = today["raw_score"].rank(pct=True)

# =========================
# 🔥 改良①：極端値除外（安定化）
# =========================
today = today[(today["score"] > 0.8)]  # 上位20%だけ残す

# =========================
# 🔥 改良②：最終選択
# =========================
today = today.sort_values("score", ascending=False).head(TOP_N)

today["PredRank"] = range(1, len(today)+1)

# =========================
# 🔥 改良③：ポジションサイズ用（重要）
# =========================
# トレンドをそのまま利用
today["weight"] = 1 + today["Trend_5_z"].clip(-1, 1)

# 正規化（合計1）
today["weight"] = today["weight"] / today["weight"].sum()

# =========================
# 出力
# =========================
print("\n=== 今日の銘柄（最終版） ===")
print(today[[
    "Ticker","Name",
    "raw_score","score",
    "Trend_5_z","weight",
    "PredRank"
]])

# =========================
# デバッグ（重要）
# =========================
print("\n=== SCORE分布 ===")
print(today["score"].describe())

print("\n=== 使用特徴量数 ===")
print(len(FEATURES))
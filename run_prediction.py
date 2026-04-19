import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRanker

# =========================
# 設定（4日戦略 AI主体）
# =========================
TOP_N = 3
TOP_RATE = 0.02

BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_4d.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest_4d.parquet")

# =========================
# 特徴量
# =========================
FEATURES = [
    "Breakout",
    "Volume_Spike",
    "Vol_Expansion",
    "Gap",
    "final_score"
]

# =========================
# =========================
# 🔥 ① 学習
# =========================
# =========================

train_df = pd.read_parquet(TRAIN_DATA_PATH).copy()

missing = [c for c in FEATURES + ["TargetRank", "Date"] if c not in train_df.columns]
if missing:
    raise ValueError(f"Missing columns in train: {missing}")

train_df = train_df.dropna(subset=FEATURES + ["TargetRank"])

X = train_df[FEATURES]
y = train_df["TargetRank"]

# group（ランキング学習に必須）
group = train_df.groupby("Date").size().values

model = LGBMRanker(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X, y, group=group)

print("✅ モデル学習完了")

# =========================
# =========================
# 🔥 ② 予測
# =========================
# =========================

df = pd.read_parquet(PREDICT_DATA_PATH).copy()

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in predict: {missing}")

df = df.dropna(subset=FEATURES).copy()

# =========================
# 🔥 AI予測
# =========================
df["pred_score"] = model.predict(df[FEATURES])

# =========================
# 🔥 ランク化
# =========================
df["pred_rank"] = df["pred_score"].rank(ascending=False, pct=True)

# =========================
# 🔥 上位フィルター
# =========================
df = df[df["pred_rank"] <= TOP_RATE].copy()

if len(df) == 0:
    print("⚠️ 条件満たす銘柄なし")
    exit()

# =========================
# 🔥 最終選定
# =========================
df = df.sort_values("pred_score", ascending=False).head(TOP_N).copy()
df["rank"] = range(1, len(df) + 1)

# =========================
# 🔥 weight
# =========================
df["weight"] = np.exp(df["pred_score"])
df["weight"] /= df["weight"].sum()

# =========================
# 出力
# =========================
print("\n=== 今日の銘柄（AI主体モデル） ===")

print(df[[
    "Ticker",
    "pred_score",
    "Breakout",
    "Volume_Spike",
    "Vol_Expansion",
    "Gap",
    "weight",
    "rank"
]])
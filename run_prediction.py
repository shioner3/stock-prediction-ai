import pandas as pd
import numpy as np
import os
import pickle

# =========================
# 設定（4日戦略 AI主体）
# =========================
TOP_N = 3
TOP_RATE = 0.02

BASE_DIR = os.path.dirname(__file__)

PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest_4d.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# =========================
# データ読み込み
# =========================
df = pd.read_parquet(PREDICT_DATA_PATH).copy()

# =========================
# モデル読み込み
# =========================
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# =========================
# 特徴量
# =========================
FEATURES = [
    "Breakout",
    "Volume_Spike",
    "Vol_Expansion",
    "Gap"
]

# =========================
# 欠損チェック
# =========================
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=FEATURES).copy()

# =========================
# 🔥 予測（AI主体）
# =========================
df["pred_score"] = model.predict(df[FEATURES])

# =========================
# 🔥 ランク化（クロスセクション）
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
# 🔥 最終選定（TOP_N）
# =========================
df = df.sort_values("pred_score", ascending=False).head(TOP_N).copy()
df["rank"] = range(1, len(df) + 1)

# =========================
# 🔥 weight（超重要）
# =========================
# スコアを強調（差を広げる）
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
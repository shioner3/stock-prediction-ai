import pandas as pd
import numpy as np
import duckdb

# =========================
# 設定
# =========================
PARQUET_FILE = "stock_data/prices.parquet"

TRAIN_SAVE_PATH = "ml_dataset_7d.parquet"
PREDICT_SAVE_PATH = "ml_dataset_latest_7d.parquet"

HOLD_DAYS = 7
MIN_COUNT = 3000
MIN_PRICE = 50
TARGET_CLIP = 0.5

# =========================
# データ読み込み
# =========================
con = duckdb.connect()

df = con.execute(f"""
SELECT Date, Ticker, Open, High, Low, Close, Volume
FROM '{PARQUET_FILE}'
""").df()

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# =========================
# フィルタ
# =========================
df = df[df["Close"] > MIN_PRICE].copy()

counts = df["Date"].value_counts()
valid_dates = counts[counts >= MIN_COUNT].index
df = df[df["Date"].isin(valid_dates)].copy()

# =========================
# 🔥 基本リターン
# =========================
df["Return_5"] = df.groupby("Ticker")["Close"].pct_change(5).shift(1)

# =========================
# 🔥 ボラ
# =========================
df["Volatility"] = df.groupby("Ticker")["Close"].pct_change().shift(1).rolling(20).std()

# =========================
# 🔥 出来高Z
# =========================
vol_mean = df.groupby("Ticker")["Volume"].transform(lambda x: x.shift(1).rolling(20).mean())
vol_std  = df.groupby("Ticker")["Volume"].transform(lambda x: x.shift(1).rolling(20).std())

df["Volume_Z"] = (df["Volume"].shift(1) - vol_mean) / (vol_std + 1e-9)

# =========================
# 🔥 【核心】爆発検出特徴
# =========================

# ① ブレイクアウト
rolling_high = df.groupby("Ticker")["High"].transform(
    lambda x: x.shift(1).rolling(20).max()
)
df["Breakout"] = df["Close"].shift(1) / (rolling_high + 1e-9)

# ② 出来高スパイク
df["Volume_Spike"] = df["Volume_Z"] * df["Return_5"]

# ③ ボラ拡大
df["Vol_Expansion"] = df["Volatility"] * df["Return_5"].abs()

# ④ ギャップ
df["Gap"] = df["Open"] / df["Close"].shift(1) - 1

# =========================
# 🔥 正規化（超重要）
# =========================
for col in ["Breakout", "Volume_Spike", "Vol_Expansion", "Gap"]:
    df[col] = df.groupby("Date")[col].transform(
        lambda x: np.clip(
            (x - x.mean()) / (x.std() + 1e-9),
            -3, 3
        )
    )

# =========================
# 🔥 最終スコア（爆発特化）
# =========================
df["final_score"] = (
    0.40 * df["Breakout"]
    + 0.25 * df["Volume_Spike"]
    + 0.20 * df["Vol_Expansion"]
    + 0.15 * df["Gap"]
)

# =========================
# 🎯 Target（爆発用）
# =========================
future_max = df.groupby("Ticker")["High"].shift(-1).rolling(HOLD_DAYS).max()

df["Target"] = future_max / df["Close"] - 1
df["Target"] = df["Target"].clip(-TARGET_CLIP, TARGET_CLIP)

df["TargetRank"] = df.groupby("Date")["Target"].rank(pct=True)

# =========================
# FEATURES
# =========================
FEATURES = [
    "Breakout",
    "Volume_Spike",
    "Vol_Expansion",
    "Gap"
]

# =========================
# データ作成
# =========================
train_df = df.dropna(subset=FEATURES + ["TargetRank"]).reset_index(drop=True)

latest_date = df["Date"].max()
predict_df = df[df["Date"] == latest_date].dropna(subset=FEATURES).reset_index(drop=True)

# =========================
# 保存
# =========================
train_df.to_parquet(TRAIN_SAVE_PATH)
predict_df.to_parquet(PREDICT_SAVE_PATH)

print("\n🔥 保存完了（爆発検出モデル）")
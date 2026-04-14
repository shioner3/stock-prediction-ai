import pandas as pd
import numpy as np
import duckdb

# =========================
# 設定（15日専用）
# =========================
PARQUET_FILE = "stock_data/prices.parquet"

TRAIN_SAVE_PATH = "ml_dataset_15d.parquet"
PREDICT_SAVE_PATH = "ml_dataset_latest_15d.parquet"

HOLD_DAYS = 15
MIN_COUNT = 3000
Z_WINDOW = 40   # ← 長めに変更

MIN_PRICE = 50
TARGET_CLIP = 0.5   # ← 15日なので広げる

# =========================
# データ読み込み
# =========================
con = duckdb.connect()

df = con.execute(f"""
SELECT 
    Date,
    Ticker,
    Open,
    High,
    Low,
    Close,
    Volume
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
# 🔥 中期リターン（追加）
# =========================
df["Return_5"]  = df.groupby("Ticker")["Close"].pct_change(5).shift(1)
df["Return_10"] = df.groupby("Ticker")["Close"].pct_change(10).shift(1)
df["Return_20"] = df.groupby("Ticker")["Close"].pct_change(20).shift(1)

# =========================
# 🔥 移動平均（長め）
# =========================
for w in [5, 10, 20, 30]:
    ma = df.groupby("Ticker")["Close"].transform(
        lambda x: x.shift(1).rolling(w).mean()
    )
    df[f"MA{w}_ratio"] = df["Close"].shift(1) / (ma + 1e-9)

# =========================
# ボラティリティ（長め）
# =========================
df["Volatility"] = df.groupby("Ticker")["Close"].pct_change().transform(
    lambda x: x.rolling(20).std()
)

# =========================
# トレンド（強化版）
# =========================
df["Trend_10"] = df.groupby("Ticker")["Close"].shift(1) / df.groupby("Ticker")["Close"].shift(11) - 1
df["Trend_20"] = df.groupby("Ticker")["Close"].shift(1) / df.groupby("Ticker")["Close"].shift(21) - 1
df["Trend_40"] = df.groupby("Ticker")["Close"].shift(1) / df.groupby("Ticker")["Close"].shift(41) - 1

def zscore(x):
    return (x - x.rolling(Z_WINDOW).mean()) / (x.rolling(Z_WINDOW).std() + 1e-9)

df["Trend_10_z"] = df.groupby("Ticker")["Trend_10"].transform(zscore)
df["Trend_20_z"] = df.groupby("Ticker")["Trend_20"].transform(zscore)
df["Trend_40_z"] = df.groupby("Ticker")["Trend_40"].transform(zscore)

# =========================
# ドローダウン（押し目）
# =========================
rolling_max_20 = df.groupby("Ticker")["Close"].transform(
    lambda x: x.shift(1).rolling(20).max()
)
rolling_max_40 = df.groupby("Ticker")["Close"].transform(
    lambda x: x.shift(1).rolling(40).max()
)

df["DD_20"] = df["Close"].shift(1) / (rolling_max_20 + 1e-9) - 1
df["DD_40"] = df["Close"].shift(1) / (rolling_max_40 + 1e-9) - 1

# =========================
# トレンド効率（超重要）
# =========================
df["TrendVol"] = df["Trend_20"] / (df["Volatility"] + 1e-9)

# =========================
# 出来高（中期）
# =========================
vol_mean = df.groupby("Ticker")["Volume"].transform(
    lambda x: x.shift(1).rolling(20).mean()
)
vol_std  = df.groupby("Ticker")["Volume"].transform(
    lambda x: x.shift(1).rolling(20).std()
)

df["Volume_Z"] = (df["Volume"].shift(1) - vol_mean) / (vol_std + 1e-9)

# =========================
# 市場特徴量
# =========================
market_mean = df.groupby("Date")["Return_5"].transform("mean")
market_std  = df.groupby("Date")["Return_5"].transform("std")

df["Market_Z"] = (df["Return_5"] - market_mean) / (market_std + 1e-9)
df["Market_Trend"] = df.groupby("Date")["Trend_20"].transform("mean")

# =========================
# クロスセクション
# =========================
rank_cols = [
    "Return_10",
    "Trend_20_z",
    "TrendVol",
    "DD_20"
]

for col in rank_cols:
    df[f"{col}_rank"] = df.groupby("Date")[col].rank(pct=True)

# =========================
# 🎯 Target（15日）
# =========================
df["Target"] = (
    df.groupby("Ticker")["Close"].shift(-HOLD_DAYS) / df["Close"] - 1
)

df["Target"] = df["Target"].clip(-TARGET_CLIP, TARGET_CLIP)

# =========================
# FEATURES（15日専用）
# =========================
FEATURES = [
    "Return_5","Return_10","Return_20",
    "MA5_ratio","MA10_ratio","MA20_ratio","MA30_ratio",
    "Volatility",
    "Trend_10_z","Trend_20_z","Trend_40_z",
    "DD_20","DD_40",
    "TrendVol","Volume_Z",
    "Return_10_rank","Trend_20_z_rank",
    "TrendVol_rank","DD_20_rank",
    "Market_Z","Market_Trend"
]

# =========================
# データ作成
# =========================
train_df = df.dropna(subset=FEATURES + ["Target"]).reset_index(drop=True)

latest_date = df["Date"].max()
predict_df = df[df["Date"] == latest_date].dropna(subset=FEATURES).reset_index(drop=True)

# =========================
# 保存
# =========================
train_df.to_parquet(TRAIN_SAVE_PATH)
predict_df.to_parquet(PREDICT_SAVE_PATH)

print("\n保存完了（15日ホールド専用）")
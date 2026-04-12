import pandas as pd
import numpy as np
import duckdb

# =========================
# 設定
# =========================
PARQUET_FILE = "stock_data/prices.parquet"

TRAIN_SAVE_PATH = "ml_dataset.parquet"
PREDICT_SAVE_PATH = "ml_dataset_latest.parquet"

HOLD_DAYS = 3
MIN_COUNT = 3000
Z_WINDOW = 20

MIN_PRICE = 50
TARGET_CLIP = 0.3

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
# 基本特徴量（リーク防止）
# =========================
df["Return_1"] = df.groupby("Ticker")["Close"].pct_change().shift(1)
df["Return_3"] = df.groupby("Ticker")["Close"].pct_change(3).shift(1)

for w in [3, 5, 10]:
    ma = df.groupby("Ticker")["Close"].transform(
        lambda x: x.shift(1).rolling(w).mean()
    )
    df[f"MA{w}_ratio"] = df["Close"].shift(1) / ma

df["Volatility"] = df.groupby("Ticker")["Return_1"].transform(
    lambda x: x.rolling(10).std()
)

df["Volume_change"] = df.groupby("Ticker")["Volume"].pct_change().shift(1)

vol_ma5 = df.groupby("Ticker")["Volume"].transform(
    lambda x: x.shift(1).rolling(5).mean()
)
df["Volume_ratio"] = df["Volume"].shift(1) / (vol_ma5 + 1e-9)

df["HL_range"] = (
    (df["High"].shift(1) - df["Low"].shift(1)) /
    (df.groupby("Ticker")["Close"].shift(1) + 1e-9)
)

# =========================
# 市場特徴量
# =========================
market_mean = df.groupby("Date")["Return_1"].transform("mean")
market_std  = df.groupby("Date")["Return_1"].transform("std")

df["Rel_Return_1"] = df["Return_1"] - market_mean
df["Market_Z"] = (df["Return_1"] - market_mean) / (market_std + 1e-9)

# =========================
# トレンド
# =========================
df["Trend_5"] = (
    df.groupby("Ticker")["Close"].shift(1) /
    df.groupby("Ticker")["Close"].shift(6) - 1
)

df["Trend_10"] = (
    df.groupby("Ticker")["Close"].shift(1) /
    df.groupby("Ticker")["Close"].shift(11) - 1
)

trend5_mean = df.groupby("Ticker")["Trend_5"].transform(
    lambda x: x.rolling(Z_WINDOW).mean()
)
trend5_std  = df.groupby("Ticker")["Trend_5"].transform(
    lambda x: x.rolling(Z_WINDOW).std()
)
df["Trend_5_z"] = (df["Trend_5"] - trend5_mean) / (trend5_std + 1e-9)

trend10_mean = df.groupby("Ticker")["Trend_10"].transform(
    lambda x: x.rolling(Z_WINDOW).mean()
)
trend10_std  = df.groupby("Ticker")["Trend_10"].transform(
    lambda x: x.rolling(Z_WINDOW).std()
)
df["Trend_10_z"] = (df["Trend_10"] - trend10_mean) / (trend10_std + 1e-9)

df["Trend_diff"] = df["Trend_5"] - df["Trend_10"]

# =========================
# 追加特徴量
# =========================
df["Gap"] = (
    df["Open"].shift(1) /
    df.groupby("Ticker")["Close"].shift(2) - 1
)

df["Volatility_change"] = (
    df["Volatility"] /
    (df.groupby("Ticker")["Volatility"].shift(5) + 1e-9)
)

df["Momentum_acc"] = df["Return_1"] - df["Return_3"]

rolling_max_5 = df.groupby("Ticker")["Close"].transform(
    lambda x: x.shift(1).rolling(5).max()
)
rolling_max_10 = df.groupby("Ticker")["Close"].transform(
    lambda x: x.shift(1).rolling(10).max()
)

df["DD_5"] = df["Close"].shift(1) / (rolling_max_5 + 1e-9) - 1
df["DD_10"] = df["Close"].shift(1) / (rolling_max_10 + 1e-9) - 1

df["TrendVol"] = df["Trend_5"] / (df["Volatility"] + 1e-9)

vol_mean = df.groupby("Ticker")["Volume"].transform(
    lambda x: x.shift(1).rolling(20).mean()
)
vol_std  = df.groupby("Ticker")["Volume"].transform(
    lambda x: x.shift(1).rolling(20).std()
)

df["Volume_Z"] = (
    (df["Volume"].shift(1) - vol_mean) /
    (vol_std + 1e-9)
)

# =========================
# クロスセクション（厳選）
# =========================
rank_cols = [
    "Return_1",
    "Volume_ratio",
    "Trend_5_z",
    "TrendVol"
]

for col in rank_cols:
    df[f"{col}_rank"] = df.groupby("Date")[col].rank(pct=True)

df["Market_Trend"] = df.groupby("Date")["Trend_5"].transform("mean")

# =========================
# 🎯 Target（HOLD一致）
# =========================
df["Target"] = (
    df.groupby("Ticker")["Close"].shift(-HOLD_DAYS) / df["Close"] - 1
)

df["Target"] = df["Target"].clip(-TARGET_CLIP, TARGET_CLIP)

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
    "Gap","Volatility_change","Momentum_acc",
    "DD_5","DD_10",
    "TrendVol","Volume_Z",
    "Return_1_rank","Volume_ratio_rank",
    "Trend_5_z_rank","TrendVol_rank",
    "Market_Z","Market_Trend"
]

# =========================
# データ作成
# =========================
train_df = df.dropna(subset=FEATURES + ["Target"]).reset_index(drop=True)

latest_date = df["Date"].max()
predict_df = df[df["Date"] == latest_date].dropna(subset=FEATURES).reset_index(drop=True)

# =========================
# デバッグ
# =========================
print("\n=== DATA VALIDATION ===")

print("\nNaN率（上位10）:")
print(df.isna().mean().sort_values(ascending=False).head(10))

print("\nTarget stats:")
print(df["Target"].describe())

print("\n銘柄数:", df["Ticker"].nunique())
print("日付数:", df["Date"].nunique())

print("\n=== TARGET CHECK ===")
print(train_df["Target"].describe())

print("\n=== DATE CHECK ===")
print("Train:", train_df["Date"].min(), "→", train_df["Date"].max())
print("Predict:", latest_date)

# =========================
# 保存
# =========================
train_df.to_parquet(TRAIN_SAVE_PATH)
predict_df.to_parquet(PREDICT_SAVE_PATH)

print("\n保存完了（プロ仕様版）")
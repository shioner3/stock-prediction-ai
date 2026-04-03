import pandas as pd
import numpy as np
import duckdb

# =========================
# 設定
# =========================
PARQUET_FILE = "stock_data/prices.parquet"

TRAIN_SAVE_PATH = "ml_dataset.parquet"
PREDICT_SAVE_PATH = "ml_dataset_latest.parquet"

HOLD_DAYS = 7
MIN_COUNT = 3000

Z_WINDOW = 20  # ★Z-scoreの基準期間

# =========================
# データ読み込み
# =========================
con = duckdb.connect()

df = con.execute(f"""
SELECT 
    Date,
    Ticker,
    Name,
    Open,
    High,
    Low,
    Close,
    Volume
FROM '{PARQUET_FILE}'
""").df()

print("元データサイズ:", df.shape)

df["Date"] = pd.to_datetime(df["Date"])

# =========================
# ソート
# =========================
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# =========================
# 日付フィルター
# =========================
counts = df["Date"].value_counts()
valid_dates = counts[counts >= MIN_COUNT].index
df = df[df["Date"].isin(valid_dates)].copy()

print("フィルタ後サイズ:", df.shape)

# =========================
# 基本特徴量
# =========================
df["Return_1"] = df.groupby("Ticker")["Close"].pct_change()
df["Return_3"] = df.groupby("Ticker")["Close"].pct_change(3)

df["MA3"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(3).mean())
df["MA5"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(5).mean())
df["MA10"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(10).mean())

df["MA3_ratio"] = df["Close"] / df["MA3"]
df["MA5_ratio"] = df["Close"] / df["MA5"]
df["MA10_ratio"] = df["Close"] / df["MA10"]

df["Volatility"] = df.groupby("Ticker")["Return_1"].transform(lambda x: x.rolling(10).std())

df["Volume_change"] = df.groupby("Ticker")["Volume"].pct_change()
df["Volume_ma5"] = df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(5).mean())
df["Volume_ratio"] = df["Volume"] / df["Volume_ma5"]

df["HL_range"] = (df["High"] - df["Low"]) / df["Close"]

# =========================
# 市場特徴
# =========================
df["Market_Return_1"] = df.groupby("Date")["Return_1"].transform("mean")

df["Rel_Return_1"] = df["Return_1"] - df["Market_Return_1"]

df["Market_Vol"] = df.groupby("Date")["Volatility"].transform("mean")
df["Vol_Ratio"] = df["Volatility"] / df["Market_Vol"]

# =========================
# Trend（生値）
# =========================
df["Trend_5"] = df["Close"] / df.groupby("Ticker")["Close"].shift(5) - 1
df["Trend_10"] = df["Close"] / df.groupby("Ticker")["Close"].shift(10) - 1

# =========================
# 🔥 Z-score化（核心）
# =========================
def zscore(group, col, window):
    mean = group[col].rolling(window).mean()
    std = group[col].rolling(window).std()
    return (group[col] - mean) / (std + 1e-9)

# Trend系
df["Trend_5_z"] = df.groupby("Ticker", group_keys=False).apply(
    lambda x: zscore(x, "Trend_5", Z_WINDOW)
)

df["Trend_10_z"] = df.groupby("Ticker", group_keys=False).apply(
    lambda x: zscore(x, "Trend_10", Z_WINDOW)
)

# Volatility
df["Volatility_z"] = df.groupby("Ticker", group_keys=False).apply(
    lambda x: zscore(x, "Volatility", Z_WINDOW)
)

# Volume
df["Volume_ratio_z"] = df.groupby("Ticker", group_keys=False).apply(
    lambda x: zscore(x, "Volume_ratio", Z_WINDOW)
)

# Market系（重要）
df["Market_Return_z"] = df.groupby("Date")["Market_Return_1"].transform(
    lambda x: (x - x.rolling(20).mean()) / (x.rolling(20).std() + 1e-9)
)

# =========================
# RSI
# =========================
delta = df.groupby("Ticker")["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.groupby(df["Ticker"]).transform(lambda x: x.rolling(7).mean())
avg_loss = loss.groupby(df["Ticker"]).transform(lambda x: x.rolling(7).mean())

rs = avg_gain / (avg_loss + 1e-9)
df["RSI"] = 100 - (100 / (1 + rs))

# =========================
# ストップ高
# =========================
df["limit_up_raw"] = (df["Return_1"] > 0.15).astype(int)
df["limit_up_flag"] = df.groupby("Ticker")["limit_up_raw"].shift(1).fillna(0)

# =========================
# Target
# =========================
df["FutureReturn"] = (
    df.groupby("Ticker")["Close"].shift(-HOLD_DAYS) / df["Close"] - 1
)

df["Target"] = (df["FutureReturn"] > 0).astype(int)

df = df.dropna(subset=["FutureReturn"])

# =========================
# 無限値処理
# =========================
df = df.replace([np.inf, -np.inf], np.nan)

# =========================
# FEATURES（Z-score追加）
# =========================
FEATURES = [
    "Return_1","Return_3",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio",
    "HL_range",
    "RSI",
    "Rel_Return_1",
    "Trend_5_z",
    "Trend_10_z",
    "Volatility_z",
    "Volume_ratio_z",
    "Market_Return_z"
]

# =========================
# 学習データ
# =========================
train_df = df.dropna(subset=FEATURES + ["Target"]).reset_index(drop=True)

# =========================
# 予測データ
# =========================
latest_date = df["Date"].max()

predict_df = df[df["Date"] == latest_date].dropna(subset=FEATURES).reset_index(drop=True)

# =========================
# デバッグ
# =========================
print("\n=== TRAIN DEBUG ===")
print("rows:", len(train_df))
print("Target mean:", train_df["Target"].mean())

print("\n=== PREDICT DEBUG ===")
print("rows:", len(predict_df))

# =========================
# 保存
# =========================
train_df.to_parquet(TRAIN_SAVE_PATH)
predict_df.to_parquet(PREDICT_SAVE_PATH)

print("\n保存完了")
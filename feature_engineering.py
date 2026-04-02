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

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# =========================
# 日付フィルター
# =========================
counts = df["Date"].value_counts()
valid_dates = counts[counts >= MIN_COUNT].index
df = df[df["Date"].isin(valid_dates)].copy()

# =========================
# 基本特徴量
# =========================
df["Return_1"] = df.groupby("Ticker")["Close"].pct_change()
df["Return_3"] = df.groupby("Ticker")["Close"].pct_change(3)
df["Return_5"] = df.groupby("Ticker")["Close"].pct_change(5)

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
# 強化特徴量
# =========================
df["EMA5"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=5).mean())
df["EMA20"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=20).mean())
df["EMA_gap"] = (df["EMA5"] - df["EMA20"]) / df["Close"]

df["Momentum_5"] = df.groupby("Ticker")["Close"].pct_change(5)
df["Momentum_10"] = df.groupby("Ticker")["Close"].pct_change(10)

df["Volume_ma10"] = df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(10).mean())
df["Volume_accel"] = df["Volume"] / df["Volume_ma10"]

df["TR"] = df["High"] - df["Low"]
df["ATR"] = df.groupby("Ticker")["TR"].transform(lambda x: x.rolling(14).mean())
df["ATR_ratio"] = df["ATR"] / df["Close"]

# =========================
# 🔥 Target（元に戻す）
# =========================
df["FutureReturn"] = (
    df.groupby("Ticker")["Close"].shift(-HOLD_DAYS) / df["Close"] - 1
)

threshold = df["FutureReturn"].groupby(df["Date"]).transform(
    lambda x: x.quantile(0.7)
)

# ✅ 相対だけに戻す
df["Target"] = (df["FutureReturn"] > threshold).astype(int)

df = df.dropna(subset=["FutureReturn"])

# =========================
# 特徴量
# =========================
FEATURES = [
    "Return_1","Return_3","Return_5",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio","Volume_accel",
    "HL_range",
    "EMA_gap",
    "Momentum_5","Momentum_10",
    "ATR_ratio"
]

# =========================
# 保存
# =========================
train_df = df.dropna(subset=FEATURES + ["Target"]).reset_index(drop=True)

latest_date = df["Date"].max()
predict_df = df[df["Date"] == latest_date].dropna(subset=FEATURES).reset_index(drop=True)

train_df.to_parquet(TRAIN_SAVE_PATH)
predict_df.to_parquet(PREDICT_SAVE_PATH)

print("保存完了")
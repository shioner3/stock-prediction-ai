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
Z_WINDOW = 20

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
# 特徴量
# =========================
df["Return_1"] = df.groupby("Ticker")["Close"].pct_change().shift(1)
df["Return_3"] = df.groupby("Ticker")["Close"].pct_change(3).shift(1)

df["Rank_Return_1"] = df.groupby("Date")["Return_1"].rank(pct=True)
df["Rank_Volume"] = df.groupby("Date")["Volume"].rank(pct=True)

# MA
for w in [3,5,10]:
    df[f"MA{w}"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(w).mean()).shift(1)
    df[f"MA{w}_ratio"] = df["Close"].shift(1) / df[f"MA{w}"]

# Volatility
df["Volatility"] = df.groupby("Ticker")["Return_1"].transform(lambda x: x.rolling(10).std()).shift(1)

# Volume
df["Volume_change"] = df.groupby("Ticker")["Volume"].pct_change().shift(1)
df["Volume_ma5"] = df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(5).mean()).shift(1)
df["Volume_ratio"] = df["Volume"].shift(1) / df["Volume_ma5"]

# HL
df["HL_range"] = ((df["High"] - df["Low"]) / df["Close"]).shift(1)

# Market
df["Market_Return_1"] = df.groupby("Date")["Return_1"].transform("mean")
df["Rel_Return_1"] = df["Return_1"] - df["Market_Return_1"]

# Trend
df["Trend_5"] = df["Close"].shift(1) / df.groupby("Ticker")["Close"].shift(6) - 1
df["Trend_10"] = df["Close"].shift(1) / df.groupby("Ticker")["Close"].shift(11) - 1

# Z
def zscore(group, col, window):
    return (group[col] - group[col].rolling(window).mean()) / (group[col].rolling(window).std() + 1e-9)

df["Trend_5_z"] = df.groupby("Ticker", group_keys=False).apply(lambda x: zscore(x, "Trend_5", Z_WINDOW))
df["Trend_10_z"] = df.groupby("Ticker", group_keys=False).apply(lambda x: zscore(x, "Trend_10", Z_WINDOW))

# =========================
# 🎯 回帰ターゲット
# =========================
df["Target"] = df.groupby("Ticker")["Close"].shift(-HOLD_DAYS) / df["Close"] - 1

df = df.dropna(subset=["Target"])

# =========================
# FEATURES
# =========================
FEATURES = [
    "Return_1","Return_3",
    "Rank_Return_1","Rank_Volume",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio",
    "HL_range",
    "Rel_Return_1",
    "Trend_5_z","Trend_10_z"
]

train_df = df.dropna(subset=FEATURES + ["Target"])
predict_df = df[df["Date"] == df["Date"].max()].dropna(subset=FEATURES)

train_df.to_parquet(TRAIN_SAVE_PATH)
predict_df.to_parquet(PREDICT_SAVE_PATH)

print("保存完了")
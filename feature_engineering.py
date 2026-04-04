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
# 基本特徴量（全部shift）
# =========================
df["Return_1"] = df.groupby("Ticker")["Close"].pct_change().shift(1)
df["Return_3"] = df.groupby("Ticker")["Close"].pct_change(3).shift(1)

# 🔥 Rank追加（超重要）
df["Rank_Return_1"] = df.groupby("Date")["Return_1"].rank(pct=True)

# MA
df["MA3"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(3).mean()).shift(1)
df["MA5"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(5).mean()).shift(1)
df["MA10"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(10).mean()).shift(1)

df["MA3_ratio"] = df["Close"].shift(1) / df["MA3"]
df["MA5_ratio"] = df["Close"].shift(1) / df["MA5"]
df["MA10_ratio"] = df["Close"].shift(1) / df["MA10"]

# Volatility
df["Volatility"] = df.groupby("Ticker")["Return_1"].transform(lambda x: x.rolling(10).std()).shift(1)

# Volume
df["Volume_change"] = df.groupby("Ticker")["Volume"].pct_change().shift(1)
df["Volume_ma5"] = df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(5).mean()).shift(1)
df["Volume_ratio"] = df["Volume"].shift(1) / df["Volume_ma5"]

# 高値安値
df["HL_range"] = ((df["High"] - df["Low"]) / df["Close"]).shift(1)

# =========================
# 市場特徴（リーク防止）
# =========================
df["Market_Return_1"] = df.groupby("Date")["Return_1"].transform("mean")

df["Rel_Return_1"] = df["Return_1"] - df["Market_Return_1"]

df["Market_Vol"] = df.groupby("Date")["Volatility"].transform("mean")
df["Vol_Ratio"] = df["Volatility"] / df["Market_Vol"]

# =========================
# Trend
# =========================
df["Trend_5"] = df["Close"].shift(1) / df.groupby("Ticker")["Close"].shift(6) - 1
df["Trend_10"] = df["Close"].shift(1) / df.groupby("Ticker")["Close"].shift(11) - 1

# =========================
# Z-score
# =========================
def zscore(group, col, window):
    mean = group[col].rolling(window).mean()
    std = group[col].rolling(window).std()
    return (group[col] - mean) / (std + 1e-9)

df["Trend_5_z"] = df.groupby("Ticker", group_keys=False).apply(lambda x: zscore(x, "Trend_5", Z_WINDOW))
df["Trend_10_z"] = df.groupby("Ticker", group_keys=False).apply(lambda x: zscore(x, "Trend_10", Z_WINDOW))
df["Volatility_z"] = df.groupby("Ticker", group_keys=False).apply(lambda x: zscore(x, "Volatility", Z_WINDOW))
df["Volume_ratio_z"] = df.groupby("Ticker", group_keys=False).apply(lambda x: zscore(x, "Volume_ratio", Z_WINDOW))

# Market Z
df["Market_Return_z"] = df.groupby("Date")["Market_Return_1"].transform(
    lambda x: (x - x.rolling(20).mean()) / (x.rolling(20).std() + 1e-9)
)

# =========================
# RSI（shift）
# =========================
delta = df.groupby("Ticker")["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.groupby(df["Ticker"]).transform(lambda x: x.rolling(7).mean()).shift(1)
avg_loss = loss.groupby(df["Ticker"]).transform(lambda x: x.rolling(7).mean()).shift(1)

rs = avg_gain / (avg_loss + 1e-9)
df["RSI"] = 100 - (100 / (1 + rs))

# =========================
# ストップ高
# =========================
df["limit_up_raw"] = (df["Return_1"] > 0.15).astype(int)
df["limit_up_flag"] = df.groupby("Ticker")["limit_up_raw"].shift(1).fillna(0)

# =========================
# Target（🔥強化版）
# =========================
df["FutureReturn"] = (
    df.groupby("Ticker")["Close"].shift(-HOLD_DAYS) / df["Close"] - 1
)

# 🔥 ここ変更
df["Target"] = (df["FutureReturn"] > 0.03).astype(int)

df = df.dropna(subset=["FutureReturn"])

# =========================
# 無限値処理
# =========================
df = df.replace([np.inf, -np.inf], np.nan)

# =========================
# FEATURES
# =========================
FEATURES = [
    "Return_1","Return_3",
    "Rank_Return_1",   # 🔥追加
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
print("\n=== DATE CHECK ===")
print("Train:", train_df["Date"].min(), "→", train_df["Date"].max())
print("Predict:", predict_df["Date"].unique())

print("\n=== TARGET CHECK ===")
print("Target mean:", train_df["Target"].mean())

# =========================
# 保存
# =========================
train_df.to_parquet(TRAIN_SAVE_PATH)
predict_df.to_parquet(PREDICT_SAVE_PATH)

print("\n保存完了")
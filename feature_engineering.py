import pandas as pd
import numpy as np
import duckdb
import os

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
# 🔥 強化特徴量（重要追加）
# =========================

# EMA差
df["EMA5"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=5).mean())
df["EMA20"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=20).mean())
df["EMA_gap"] = (df["EMA5"] - df["EMA20"]) / df["Close"]

# モメンタム
df["Momentum_5"] = df.groupby("Ticker")["Close"].pct_change(5)
df["Momentum_10"] = df.groupby("Ticker")["Close"].pct_change(10)

# 出来高圧力
df["Volume_ma10"] = df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(10).mean())
df["Volume_accel"] = df["Volume"] / df["Volume_ma10"]

# ボラ拡張
# TR（True Range簡易版）
df["TR"] = df["High"] - df["Low"]

df["ATR"] = df.groupby("Ticker")["TR"].transform(
    lambda x: x.rolling(14).mean()
)

df["ATR_ratio"] = df["ATR"] / df["Close"]

# =========================
# RSI（修正版）
# =========================
def rsi(series, period=7):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df["RSI"] = df.groupby("Ticker")["Close"].transform(rsi)

# =========================
# ストップ高フラグ
# =========================
df["limit_up_raw"] = (df["Return_1"] > 0.15).astype(int)
df["limit_up_flag"] = df.groupby("Ticker")["limit_up_raw"].shift(1).fillna(0)

# =========================
# 🔥 Target（強化版）
# =========================

df["FutureReturn"] = (
    df.groupby("Ticker")["Close"].shift(-HOLD_DAYS) / df["Close"] - 1
)

# 🔥 上位30%を1（重要改善）
threshold = df["FutureReturn"].groupby(df["Date"]).transform(
    lambda x: x.quantile(0.7)
)

df["Target"] = (df["FutureReturn"] > threshold).astype(int)

# 未来データ削除
df = df.dropna(subset=["FutureReturn"])

# =========================
# 無限値処理
# =========================
df = df.replace([np.inf, -np.inf], np.nan)

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
    "ATR_ratio",
    "RSI"
]

# =========================
# 学習データ
# =========================
train_df = df.dropna(subset=FEATURES + ["Target"]).copy()
train_df = train_df.reset_index(drop=True)

# =========================
# 予測データ
# =========================
latest_date = df["Date"].max()

predict_df = df[df["Date"] == latest_date].dropna(subset=FEATURES).copy()
predict_df = predict_df.reset_index(drop=True)

# =========================
# デバッグ
# =========================
print("\n=== TRAIN DEBUG ===")
print("rows:", len(train_df))
print("Target mean:", train_df["Target"].mean())
print("latest:", train_df["Date"].max())

print("\n=== PREDICT DEBUG ===")
print("rows:", len(predict_df))
print("latest:", predict_df["Date"].max())

# =========================
# 保存
# =========================
train_df.to_parquet(TRAIN_SAVE_PATH)
predict_df.to_parquet(PREDICT_SAVE_PATH)

print("\n保存完了")
print("train rows:", len(train_df))
print("predict rows:", len(predict_df))
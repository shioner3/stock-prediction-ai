import pandas as pd
import numpy as np
import os

# =========================
# パス
# =========================
INPUT_PATH = "stock_data/prices.parquet"
OUTPUT_PATH = "stock_data/features.parquet"

os.makedirs("stock_data", exist_ok=True)

# =========================
# 読み込み
# =========================
print("Loading prices...")

df = pd.read_parquet(INPUT_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# =========================
# グループ
# =========================
g = df.groupby("Ticker", group_keys=False)

# =========================
# リターン系
# =========================
df["return_1d"] = g["Close"].pct_change(1)
df["return_3d"] = g["Close"].pct_change(3)
df["return_5d"] = g["Close"].pct_change(5)

# =========================
# モメンタム
# =========================
df["momentum_5"] = g["Close"].pct_change(5)
df["momentum_10"] = g["Close"].pct_change(10)

# =========================
# 移動平均乖離
# =========================
df["ma5"] = g["Close"].transform(lambda x: x.rolling(5).mean())
df["ma20"] = g["Close"].transform(lambda x: x.rolling(20).mean())

df["ma5_diff"] = df["Close"] / df["ma5"] - 1
df["ma20_diff"] = df["Close"] / df["ma20"] - 1

# =========================
# ボラティリティ
# =========================
df["volatility_5"] = g["return_1d"].transform(lambda x: x.rolling(5).std())
df["volatility_10"] = g["return_1d"].transform(lambda x: x.rolling(10).std())

# =========================
# ATR（簡易）
# =========================
df["tr"] = np.maximum(
    df["High"] - df["Low"],
    np.maximum(
        abs(df["High"] - g["Close"].shift(1)),
        abs(df["Low"] - g["Close"].shift(1))
    )
)

df["atr_5"] = g["tr"].transform(lambda x: x.rolling(5).mean())

# =========================
# 出来高
# =========================
df["volume_ma5"] = g["Volume"].transform(lambda x: x.rolling(5).mean())
df["volume_ratio"] = df["Volume"] / df["volume_ma5"]

# =========================
# レンジ圧縮（ブレイク前検出用）
# =========================
df["high_20"] = g["High"].transform(lambda x: x.rolling(20).max())
df["low_20"] = g["Low"].transform(lambda x: x.rolling(20).min())

df["range_20"] = df["high_20"] - df["low_20"]
df["range_ratio"] = df["range_20"] / df["Close"]

# =========================
# クロスセクション（重要）
# =========================
df["return_rank"] = df.groupby("Date")["return_1d"].rank(pct=True)
df["volume_rank"] = df.groupby("Date")["volume_ratio"].rank(pct=True)
df["volatility_rank"] = df.groupby("Date")["volatility_5"].rank(pct=True)

# =========================
# 市場トレンド（簡易）
# =========================
# 市場リターン
market = (
    df.groupby("Date")["return_1d"]
    .mean()
    .rename("market_return")
)

df = df.merge(market, on="Date", how="left")

# 市場トレンド
df["market_trend_5"] = (
    df["market_return"]
    .rolling(5)
    .mean()
)

# 未来リーク防止
df["market_trend_5"] = df["market_trend_5"].shift(1)

# =========================
# シフト（未来リーク防止）
# =========================
shift_cols = [
    "return_1d", "return_3d", "return_5d",
    "momentum_5", "momentum_10",
    "ma5_diff", "ma20_diff",
    "volatility_5", "volatility_10",
    "atr_5",
    "volume_ratio",
    "range_ratio",
    "return_rank", "volume_rank", "volatility_rank",
    "market_trend_5"
]

for col in shift_cols:
    df[col] = g[col].shift(1)

# =========================
# クリーン
# =========================
df = df.dropna()

# =========================
# 保存
# =========================
df.to_parquet(OUTPUT_PATH, index=False)

print("\n=== 完了 ===")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
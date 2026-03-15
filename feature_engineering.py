import pandas as pd
import numpy as np
import duckdb

# =========================
# 設定
# =========================

DB_FILE = "stock.db"
SAVE_PATH = "ml_dataset.parquet"

HOLD_DAYS = 5

# =========================
# データ読み込み
# =========================

con = duckdb.connect(DB_FILE)

df = con.execute("""
SELECT *
FROM stock_prices
ORDER BY Ticker, Date
""").df()

print("元データサイズ:", df.shape)

# =========================
# 銘柄内特徴量
# =========================

df["Return_1"] = df.groupby("Ticker")["Close"].pct_change()

df["MA5"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(5).mean())
df["MA25"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(25).mean())
df["MA75"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(75).mean())

df["MA5_ratio"] = df["Close"] / df["MA5"]
df["MA25_ratio"] = df["Close"] / df["MA25"]
df["MA75_ratio"] = df["Close"] / df["MA75"]

df["Volatility"] = (
    df.groupby("Ticker")["Return_1"]
    .transform(lambda x: x.rolling(20).std())
)

df["Volume_change"] = df.groupby("Ticker")["Volume"].pct_change()

df["HL_range"] = (df["High"] - df["Low"]) / df["Close"]

# =========================
# RSI
# =========================

delta = df.groupby("Ticker")["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.groupby(df["Ticker"]).transform(lambda x: x.rolling(14).mean())
avg_loss = loss.groupby(df["Ticker"]).transform(lambda x: x.rolling(14).mean())

rs = avg_gain / avg_loss

df["RSI"] = 100 - (100 / (1 + rs))

# =========================
# クロスセクション特徴量
# =========================

rank_features = [
    "Return_1",
    "MA5_ratio",
    "MA25_ratio",
    "MA75_ratio",
    "Volatility",
    "Volume_change",
    "HL_range",
    "RSI"
]

for col in rank_features:

    df[col + "_rank"] = (
        df.groupby("Date")[col]
        .rank(pct=True)
    )

# =========================
# Target
# =========================

df["FutureReturn_5"] = (
    df.groupby("Ticker")["Close"].shift(-HOLD_DAYS) / df["Close"] - 1
)

df["Target"] = (
    df.groupby("Date")["FutureReturn_5"]
    .rank(pct=True)
)

# =========================
# データ整理
# =========================

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=[
    "Return_1_rank",
    "MA5_ratio_rank",
    "MA25_ratio_rank",
    "MA75_ratio_rank",
    "Volatility_rank",
    "Volume_change_rank",
    "HL_range_rank",
    "RSI_rank",
    "FutureReturn_5"
])

# =========================
# 保存
# =========================

df.to_parquet(SAVE_PATH)

print("MLデータ保存完了")
print("行数:", len(df))
print("列数:", len(df.columns))
print("最終データサイズ:", df.shape)

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

if len(df) == 0:
    print("❌ prices is empty")
    exit()

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

# =========================
# 出来高
# =========================
df["volume_ma5"] = g["Volume"].transform(lambda x: x.rolling(5).mean())
df["volume_ratio"] = df["Volume"] / df["volume_ma5"]

# =========================
# レンジ圧縮
# =========================
df["high_20"] = g["High"].transform(lambda x: x.rolling(20).max())
df["low_20"] = g["Low"].transform(lambda x: x.rolling(20).min())

df["range_20"] = df["high_20"] - df["low_20"]
df["range_ratio"] = df["range_20"] / df["Close"]

# =========================
# クロスセクション
# =========================
df["return_rank"] = df.groupby("Date")["return_1d"].rank(pct=True)
df["volume_rank"] = df.groupby("Date")["volume_ratio"].rank(pct=True)

# =========================
# 市場トレンド
# =========================
market = df.groupby("Date")["return_1d"].mean()
df["market_trend_5"] = market.rolling(5).mean()

# =========================
# シフト（未来リーク防止）
# =========================
shift_cols = [
    "return_1d",
    "return_3d",
    "ma5_diff",
    "ma20_diff",
    "volatility_5",
    "volume_ratio",
    "range_ratio",
    "return_rank",
    "volume_rank",
    "market_trend_5"
]

for col in shift_cols:
    if col == "market_trend_5":
        df[col] = df[col].shift(1)
    else:
        df[col] = g[col].shift(1)

# =========================
# 欠損処理（←ここが最重要修正）
# =========================

# rank系は0埋め
df["return_rank"] = df["return_rank"].fillna(0)
df["volume_rank"] = df["volume_rank"].fillna(0)

# marketはニュートラル扱い
df["market_trend_5"] = df["market_trend_5"].fillna(0)

# =========================
# 最小限dropna（緩くする）
# =========================
print("Before dropna:", len(df))

df = df.dropna(subset=[
    "return_3d",
    "volume_ratio"
])

print("After dropna:", len(df))

# =========================
# 不要カラム削除
# =========================
drop_cols = [
    "ma5", "ma20",
    "volume_ma5",
    "high_20", "low_20", "range_20"
]

df = df.drop(columns=drop_cols, errors="ignore")

# =========================
# 保存
# =========================
df.to_parquet(OUTPUT_PATH, index=False)

# =========================
# 完了
# =========================
print("\n=== 完了 ===")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
import pandas as pd
import numpy as np

INPUT_PATH = "stock_data/technical_features.parquet"
SAVE_PATH = "stock_data/ml_dataset.parquet"

HOLD_DAYS = 5
MARKET_TICKER = "1306"   # ★ここ重要

df = pd.read_parquet(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(["Ticker", "Date"])

print("\n===== DEBUG BEFORE =====")
print("rows:", len(df))
print("unique tickers:", df["Ticker"].nunique())

# =========================
# ★ 市場データ抽出（先にやる）
# =========================
market = df[df["Ticker"] == MARKET_TICKER][["Date", "Close"]].copy()

if len(market) == 0:
    raise ValueError("❌ MARKET DATA NOT FOUND → 1306が存在しない")

# =========================
# forward return（市場）
# =========================
market["market_return"] = (
    market["Close"].shift(-HOLD_DAYS) / market["Close"] - 1
)

market = market[["Date", "market_return"]]

# =========================
# forward return（個別株）
# =========================
df["forward_return_raw"] = (
    df.groupby("Ticker")["Close"].shift(-HOLD_DAYS) / df["Close"] - 1
)

# =========================
# ★ マージ（ここで確認）
# =========================
df = df.merge(market, on="Date", how="left")

print("\n===== DEBUG AFTER MERGE =====")
print(df[["Date", "Ticker", "market_return"]].head())
print("market_return null ratio:", df["market_return"].isna().mean())

# =========================
# ★ 市場中立化
# =========================
df["forward_return"] = df["forward_return_raw"] - df["market_return"]

# =========================
# 異常値処理
# =========================
df["forward_return"] = df["forward_return"].clip(-0.5, 0.5)
df["forward_return"] = np.log1p(df["forward_return"])

# =========================
# フィルタ
# =========================
df = df[df["Close"] > 100]
df = df[df["Volume"] > 100000]

# =========================
# target
# =========================
df["target"] = df["forward_return"]
df["label"] = (df["forward_return"] > 0).astype(int)

# =========================
# ★ dropna前にチェック
# =========================
print("\n===== BEFORE DROPNA =====")
print("rows:", len(df))
print("null stats:")
print(df[["forward_return", "market_return"]].isna().mean())

# =========================
# cleanup
# =========================
df = df.dropna()

print("\n===== AFTER DROPNA =====")
print("rows:", len(df))

# =========================
# 保存
# =========================
df.to_parquet(SAVE_PATH, index=False)

print("\nSaved:", SAVE_PATH)
print(df.shape)
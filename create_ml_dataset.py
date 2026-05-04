import pandas as pd
import numpy as np

INPUT_PATH = "stock_data/technical_features.parquet"
SAVE_PATH = "stock_data/ml_dataset.parquet"

HOLD_DAYS = 5

df = pd.read_parquet(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(["Ticker", "Date"])

# =========================
# forward return（銘柄）
# =========================
df["forward_return_raw"] = (
    df.groupby("Ticker")["Close"].shift(-HOLD_DAYS) / df["Close"] - 1
)

# =========================
# ★ 市場リターン（1306など）
# =========================
# 例：Ticker == "1306" を市場とする
market = df[df["Ticker"] == "1306"][["Date", "Close"]].copy()
market["market_return"] = (
    market["Close"].shift(-HOLD_DAYS) / market["Close"] - 1
)

market = market[["Date", "market_return"]]

# マージ
df = df.merge(market, on="Date", how="left")

# =========================
# ★ 超重要：市場中立化
# =========================
df["forward_return"] = df["forward_return_raw"] - df["market_return"]

# =========================
# 異常値除去
# =========================
df["forward_return"] = df["forward_return"].clip(-0.5, 0.5)

# =========================
# log化
# =========================
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

# =========================
# label（任意）
# =========================
df["label"] = (df["forward_return"] > 0).astype(int)

# =========================
# cleanup
# =========================
df = df.dropna()

df.to_parquet(SAVE_PATH, index=False)

print("Saved:", SAVE_PATH)
print(df.shape)
import pandas as pd
import numpy as np

INPUT_PATH = "stock_data/technical_features.parquet"
SAVE_PATH = "stock_data/ml_dataset.parquet"

HOLD_DAYS = 5

df = pd.read_parquet(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(["Ticker", "Date"])

# =========================
# forward return（正しい定義）
# =========================
df["forward_return"] = (
    df.groupby("Ticker")["Close"].shift(-HOLD_DAYS) / df["Close"] - 1
)

# =========================
# target（回帰・ランキング両対応）
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
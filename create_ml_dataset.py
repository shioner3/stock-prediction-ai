import pandas as pd
import numpy as np

INPUT_PATH = "stock_data/technical_features.parquet"
SAVE_PATH = "stock_data/ml_dataset.parquet"

HOLD_DAYS = 5

df = pd.read_parquet(INPUT_PATH)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# =========================
# forward return（最重要）
# =========================
df["forward_return"] = (
    df.groupby("Ticker")["Close"]
    .shift(-HOLD_DAYS) / df["Close"] - 1
)

# =========================
# IC用ターゲット
# =========================
df["target"] = df["forward_return"]

# =========================
# ラベル（任意）
# =========================
df["label"] = (df["forward_return"] > 0).astype(int)

# =========================
# 保存
# =========================
df = df.dropna()

df.to_parquet(SAVE_PATH, index=False)

print("Saved:", SAVE_PATH)
print(df.head())
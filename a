import pandas as pd
import numpy as np

# =========================
# 設定
# =========================
FEATURE_PATH = "stock_data/technical_features.parquet"
PRICE_PATH = "stock_data/prices.parquet"

SAVE_PATH = "stock_data/ml_dataset.parquet"

HOLD_DAYS = 5

# =========================
# データ読み込み
# =========================
print("Loading data...")

features = pd.read_parquet(FEATURE_PATH)

prices = pd.read_parquet(PRICE_PATH)

prices["Date"] = pd.to_datetime(prices["Date"])
features["Date"] = pd.to_datetime(features["Date"])

# =========================
# 必要列だけ使用
# =========================
price_cols = [
    "Date",
    "Ticker",
    "Close"
]

prices = prices[price_cols]

# =========================
# future close 作成
# =========================
print("Creating targets...")

prices = prices.sort_values(
    ["Ticker", "Date"]
)

prices["future_close"] = (
    prices.groupby("Ticker")["Close"]
    .shift(-HOLD_DAYS)
)

# =========================
# future return
# =========================
prices["target_return"] = (
    prices["future_close"]
    / prices["Close"]
    - 1
)

# =========================
# target rank
# =========================
prices["target_rank"] = (
    prices.groupby("Date")["target_return"]
    .rank(pct=True)
)

# =========================
# 必要列のみ
# =========================
target_df = prices[
    [
        "Date",
        "Ticker",
        "target_return",
        "target_rank"
    ]
]

# =========================
# merge
# =========================
print("Merging features and targets...")

df = pd.merge(
    features,
    target_df,
    on=["Date", "Ticker"],
    how="inner"
)

# =========================
# inf除去
# =========================
df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

# =========================
# 欠損除去
# =========================
df = df.dropna()

# =========================
# ソート
# =========================
df = df.sort_values(
    ["Date", "Ticker"]
).reset_index(drop=True)

# =========================
# 保存
# =========================
print("Saving ML dataset...")

df.to_parquet(
    SAVE_PATH,
    index=False
)

# =========================
# 確認
# =========================
print("Done.")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())
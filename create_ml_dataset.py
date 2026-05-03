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

features["Date"] = pd.to_datetime(
    features["Date"]
)

prices["Date"] = pd.to_datetime(
    prices["Date"]
)

# =========================
# ソート
# =========================
features = features.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)

prices = prices.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)

# =========================
# 必要列
# =========================
price_cols = [
    "Date",
    "Ticker",
    "Open",
    "Close"
]

prices = prices[price_cols]

# =========================
# 🔥 超重要
# 翌日寄りエントリー前提
#
# t日特徴量
# ↓
# t+1日寄りエントリー
# ↓
# t+1+HOLD_DAYSで売却
# =========================

# エントリー価格
prices["entry_price"] = (
    prices.groupby("Ticker")["Open"]
    .shift(-1)
)

# エグジット価格
prices["exit_price"] = (
    prices.groupby("Ticker")["Close"]
    .shift(-(HOLD_DAYS + 1))
)

# =========================
# target return
# =========================
print("Creating targets...")

prices["target_return"] = (
    prices["exit_price"]
    / prices["entry_price"]
    - 1
)

# =========================
# 異常値除去
# =========================
prices["target_return"] = (
    prices["target_return"]
    .clip(-0.50, 3.00)
)

# =========================
# target rank
# 日次クロスセクション順位
# =========================
prices["target_rank"] = (
    prices.groupby("Date")["target_return"]
    .transform(
        lambda x:
        pd.qcut(
            x,
            20,
            labels=False,
            duplicates="drop"
        )
    )
)

# =========================
# target列
# =========================
target_df = prices[
    [
        "Date",
        "Ticker",
        "entry_price",
        "exit_price",
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
# target_rank int化
# =========================
df["target_rank"] = (
    df["target_rank"]
    .astype(int)
)

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
print("\nDone.")

print("\nHead:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# =========================
# target統計
# =========================
print("\n=== Target Return Stats ===")

print(
    df["target_return"]
    .describe()
)

# =========================
# target_rank確認
# =========================
print("\n=== Target Rank Unique ===")

print(
    sorted(
        df["target_rank"]
        .unique()
    )
)
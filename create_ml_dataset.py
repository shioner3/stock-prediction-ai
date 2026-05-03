import pandas as pd
import numpy as np

INPUT_PATH = "stock_data/technical_features.parquet"
SAVE_PATH = "stock_data/ml_dataset.parquet"

HOLD_DAYS = 5

# =========================
# load
# =========================
df = pd.read_parquet(INPUT_PATH)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# =========================
# 🔥 修正①：グループ内shift（正しいforward return）
# =========================
df["future_close"] = (
    df.groupby("Ticker")["Close"]
    .shift(-HOLD_DAYS)
)

df["forward_return"] = (
    df["future_close"] / df["Close"] - 1
)

# =========================
# 🔥 修正②：日付ズレ防止（実運用必須）
# → 銘柄ごとに最終HOLD_DAYS分除去
# =========================
df = df.groupby("Ticker").apply(
    lambda x: x.iloc[:-HOLD_DAYS]
).reset_index(drop=True)

# =========================
# IC用ターゲット
# =========================
df["target"] = df["forward_return"]

# =========================
# 🔥 修正③：ノイズ減らしたラベル
# （単純0/1ではなく分位ベース推奨）
# =========================
df["label"] = (
    df.groupby("Date")["forward_return"]
    .transform(lambda x: (x > x.quantile(0.7)).astype(int))
)

# =========================
# 追加：極端値クリップ（重要）
# =========================
df["forward_return"] = df["forward_return"].clip(-0.2, 0.2)

# =========================
# final cleanup
# =========================
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# =========================
# save
# =========================
df.to_parquet(SAVE_PATH, index=False)

print("Saved:", SAVE_PATH)
print(df[["Date", "Ticker", "forward_return", "label"]].head())
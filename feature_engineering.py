import pandas as pd
import numpy as np

# =========================
# 設定
# =========================
INPUT_PATH = "stock_data/prices.parquet"
SAVE_PATH = "stock_data/technical_features.parquet"

# =========================
# データ読み込み
# =========================
print("Loading data...")

df = pd.read_parquet(INPUT_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)

# =========================
# 移動平均
# =========================
print("Calculating moving averages...")

df["ma5"] = (
    df.groupby("Ticker")["Close"]
    .transform(lambda x: x.rolling(5).mean())
)

df["ma25"] = (
    df.groupby("Ticker")["Close"]
    .transform(lambda x: x.rolling(25).mean())
)

# =========================
# close_ma_ratio
# =========================
df["close_ma5_ratio"] = (
    df["Close"] / df["ma5"]
)

df["close_ma25_ratio"] = (
    df["Close"] / df["ma25"]
)

# =========================
# ma25 slope
# =========================
df["ma25_slope"] = (
    df.groupby("Ticker")["ma25"]
    .pct_change(5, fill_method=None)
)

# =========================
# high break
# 「当日高値」を使わない
# =========================
rolling_high_20 = (
    df.groupby("Ticker")["High"]
    .transform(
        lambda x:
        x.shift(1).rolling(20).max()
    )
)

df["high_break_20d"] = (
    df["Close"] > rolling_high_20
).astype(int)

# =========================
# return
# =========================
df["return_5d"] = (
    df.groupby("Ticker")["Close"]
    .pct_change(5, fill_method=None)
)

df["return_20d"] = (
    df.groupby("Ticker")["Close"]
    .pct_change(20, fill_method=None)
)

# =========================
# relative strength
# =========================
market_return_20 = (
    df.groupby("Date")["return_20d"]
    .transform("mean")
)

df["relative_strength_20d"] = (
    df["return_20d"]
    - market_return_20
)

# =========================
# industry rs rank
# =========================
industry_mean = (
    df.groupby(
        ["Date", "Industry"]
    )["return_20d"]
    .transform("mean")
)

df["industry_rs_rank"] = (
    industry_mean.groupby(df["Date"])
    .rank(pct=True)
)

# =========================
# volume ratios
# =========================
vol_ma5 = (
    df.groupby("Ticker")["Volume"]
    .transform(lambda x: x.rolling(5).mean())
)

vol_ma20 = (
    df.groupby("Ticker")["Volume"]
    .transform(lambda x: x.rolling(20).mean())
)

df["volume_ratio_5d"] = (
    df["Volume"] / vol_ma5
)

df["volume_ratio_20d"] = (
    df["Volume"] / vol_ma20
)

# =========================
# volume zscore
# =========================
vol_std20 = (
    df.groupby("Ticker")["Volume"]
    .transform(lambda x: x.rolling(20).std())
)

df["volume_zscore"] = (
    (df["Volume"] - vol_ma20)
    / vol_std20
)

# =========================
# ATR
# =========================
prev_close = (
    df.groupby("Ticker")["Close"]
    .shift(1)
)

tr1 = (
    df["High"] - df["Low"]
)

tr2 = (
    df["High"] - prev_close
).abs()

tr3 = (
    df["Low"] - prev_close
).abs()

tr = pd.concat(
    [tr1, tr2, tr3],
    axis=1
).max(axis=1)

atr20 = (
    tr.groupby(df["Ticker"])
    .transform(lambda x: x.rolling(20).mean())
)

df["atr_ratio"] = (
    atr20 / df["Close"]
)

# =========================
# Bollinger Bands
# =========================
ma20 = (
    df.groupby("Ticker")["Close"]
    .transform(lambda x: x.rolling(20).mean())
)

std20 = (
    df.groupby("Ticker")["Close"]
    .transform(lambda x: x.rolling(20).std())
)

bb_upper = ma20 + 2 * std20
bb_lower = ma20 - 2 * std20

df["bb_width"] = (
    (bb_upper - bb_lower)
    / ma20
)

df["bb_position"] = (
    (df["Close"] - bb_lower)
    / (bb_upper - bb_lower)
)

# =========================
# range compression
# =========================
daily_range = (
    (df["High"] - df["Low"])
    / df["Close"]
)

range_ma5 = (
    daily_range.groupby(df["Ticker"])
    .transform(lambda x: x.rolling(5).mean())
)

range_ma20 = (
    daily_range.groupby(df["Ticker"])
    .transform(lambda x: x.rolling(20).mean())
)

df["range_compression_5d"] = (
    range_ma5 / range_ma20
)

# =========================
# market features
# =========================
df["nikkei_return_5d"] = (
    df.groupby("Date")["return_5d"]
    .transform("mean")
)

df["topix_trend"] = (
    df.groupby("Date")["close_ma25_ratio"]
    .transform("mean")
)

df["growth_index_strength"] = (
    df.groupby("Date")["relative_strength_20d"]
    .transform("mean")
)

# =========================
# cross sectional ranks
# =========================
df["return_rank_daily"] = (
    df.groupby("Date")["return_5d"]
    .rank(pct=True)
)

df["volume_rank_daily"] = (
    df.groupby("Date")["volume_ratio_20d"]
    .rank(pct=True)
)

df["volatility_rank"] = (
    df.groupby("Date")["atr_ratio"]
    .rank(pct=True)
)

df["rs_rank_cross_section"] = (
    df.groupby("Date")["relative_strength_20d"]
    .rank(pct=True)
)

# =========================
# upper shadow
# =========================
df["upper_shadow_ratio"] = (
    (
        df["High"]
        - np.maximum(
            df["Open"],
            df["Close"]
        )
    )
    / df["Close"]
)

# =========================
# gap up
# =========================
df["gap_up_ratio"] = (
    (df["Open"] - prev_close)
    / prev_close
)

# =========================
# 特徴量一覧
# =========================
FEATURE_COLUMNS = [

    "close_ma5_ratio",
    "close_ma25_ratio",
    "ma25_slope",
    "high_break_20d",

    "return_5d",
    "return_20d",
    "relative_strength_20d",
    "industry_rs_rank",

    "volume_ratio_5d",
    "volume_ratio_20d",
    "volume_zscore",

    "atr_ratio",
    "bb_width",
    "range_compression_5d",

    "nikkei_return_5d",
    "topix_trend",
    "growth_index_strength",

    "return_rank_daily",
    "volume_rank_daily",
    "volatility_rank",
    "rs_rank_cross_section",

    "upper_shadow_ratio",
    "gap_up_ratio",
    "bb_position"
]

# =========================
# 🔥 超重要
# 全特徴量を1日shift
# 「翌日寄りで使える情報」に統一
# =========================
print("Shifting features to avoid leakage...")

df[FEATURE_COLUMNS] = (
    df.groupby("Ticker")[FEATURE_COLUMNS]
    .shift(1)
)

# =========================
# 保存列
# =========================
SAVE_COLUMNS = [
    "Date",
    "Ticker"
] + FEATURE_COLUMNS

df = df[SAVE_COLUMNS]

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
# 保存
# =========================
print("Saving features...")

df.to_parquet(
    SAVE_PATH,
    index=False
)

# =========================
# 完了
# =========================
print("Done.")

print(df.head())

print("\nShape:")
print(df.shape)
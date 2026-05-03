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

# =========================
# 🔥 追加：列確認（重要）
# =========================
print("\n===== RAW COLUMNS =====")
print(df.columns.tolist())

print("\n===== HEAD =====")
print(df.head())

# =========================
# カラム統一（保険）
# =========================
df.columns = df.columns.str.strip()

# =========================
# 必須カラムチェック
# =========================
required_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]

missing = [c for c in required_cols if c not in df.columns]

if len(missing) > 0:
    raise ValueError(f"Missing columns: {missing}")

# =========================
# 日付処理
# =========================
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)

# =========================
# 移動平均
# =========================
print("Calculating moving averages...")

df["ma5"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(5).mean())
df["ma25"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(25).mean())

# =========================
# close_ma_ratio
# =========================
df["close_ma5_ratio"] = df["Close"] / df["ma5"]
df["close_ma25_ratio"] = df["Close"] / df["ma25"]

# =========================
# ma25 slope
# =========================
df["ma25_slope"] = df.groupby("Ticker")["ma25"].pct_change(5, fill_method=None)

# =========================
# high break
# =========================
rolling_high_20 = df.groupby("Ticker")["High"].transform(
    lambda x: x.shift(1).rolling(20).max()
)

df["high_break_20d"] = (df["Close"] > rolling_high_20).astype(int)

# =========================
# return
# =========================
df["return_5d"] = df.groupby("Ticker")["Close"].pct_change(5, fill_method=None)
df["return_20d"] = df.groupby("Ticker")["Close"].pct_change(20, fill_method=None)

# =========================
# volume ratios
# =========================
vol_ma5 = df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(5).mean())
vol_ma20 = df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(20).mean())

df["volume_ratio_5d"] = df["Volume"] / vol_ma5
df["volume_ratio_20d"] = df["Volume"] / vol_ma20

# =========================
# volume zscore
# =========================
vol_std20 = df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(20).std())
df["volume_zscore"] = (df["Volume"] - vol_ma20) / vol_std20

# =========================
# ATR
# =========================
prev_close = df.groupby("Ticker")["Close"].shift(1)

tr1 = df["High"] - df["Low"]
tr2 = (df["High"] - prev_close).abs()
tr3 = (df["Low"] - prev_close).abs()

tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

atr20 = tr.groupby(df["Ticker"]).transform(lambda x: x.rolling(20).mean())
df["atr_ratio"] = atr20 / df["Close"]

# =========================
# Bollinger Bands
# =========================
ma20 = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).mean())
std20 = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).std())

bb_upper = ma20 + 2 * std20
bb_lower = ma20 - 2 * std20

df["bb_width"] = (bb_upper - bb_lower) / ma20
df["bb_position"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower)

# =========================
# range compression
# =========================
daily_range = (df["High"] - df["Low"]) / df["Close"]

range_ma5 = daily_range.groupby(df["Ticker"]).transform(lambda x: x.rolling(5).mean())
range_ma20 = daily_range.groupby(df["Ticker"]).transform(lambda x: x.rolling(20).mean())

df["range_compression_5d"] = range_ma5 / range_ma20

# =========================
# cross sectional ranks
# =========================
df["return_rank_daily"] = df.groupby("Date")["return_5d"].rank(pct=True)
df["volume_rank_daily"] = df.groupby("Date")["volume_ratio_20d"].rank(pct=True)

# =========================
# upper shadow
# =========================
df["upper_shadow_ratio"] = (
    (df["High"] - np.maximum(df["Open"], df["Close"]))
    / df["Close"]
)

# =========================
# gap up
# =========================
df["gap_up_ratio"] = (df["Open"] - prev_close) / prev_close

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

    "volume_ratio_5d",
    "volume_ratio_20d",
    "volume_zscore",

    "atr_ratio",
    "bb_width",
    "range_compression_5d",

    "return_rank_daily",
    "volume_rank_daily",

    "upper_shadow_ratio",
    "gap_up_ratio",
    "bb_position"
]

# =========================
# leakage防止 shift
# =========================
print("Shifting features...")

df[FEATURE_COLUMNS] = df.groupby("Ticker")[FEATURE_COLUMNS].shift(1)

# =========================
# 保存
# =========================
df = df[["Date", "Ticker"] + FEATURE_COLUMNS]

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print("Saving features...")

df.to_parquet(SAVE_PATH, index=False)

print("Done.")
print(df.head())
print("\nShape:", df.shape)
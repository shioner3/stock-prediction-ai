import pandas as pd
import numpy as np
import duckdb

# =========================
# 設定（7日版）
# =========================
PARQUET_FILE = "stock_data/prices.parquet"

TRAIN_SAVE_PATH = "ml_dataset_7d.parquet"
PREDICT_SAVE_PATH = "ml_dataset_latest_7d.parquet"

HOLD_DAYS = 7
MIN_COUNT = 3000
Z_WINDOW = 40

MIN_PRICE = 50
TARGET_CLIP = 0.5

# =========================
# データ読み込み
# =========================
con = duckdb.connect()

df = con.execute(f"""
SELECT 
    Date,
    Ticker,
    Open,
    High,
    Low,
    Close,
    Volume
FROM '{PARQUET_FILE}'
""").df()

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# =========================
# フィルタ
# =========================
df = df[df["Close"] > MIN_PRICE].copy()

counts = df["Date"].value_counts()
valid_dates = counts[counts >= MIN_COUNT].index
df = df[df["Date"].isin(valid_dates)].copy()

# =========================
# 🔥 リターン
# =========================
df["Return_5"]  = df.groupby("Ticker")["Close"].pct_change(5).shift(1)
df["Return_10"] = df.groupby("Ticker")["Close"].pct_change(10).shift(1)
df["Return_20"] = df.groupby("Ticker")["Close"].pct_change(20).shift(1)

# =========================
# 🔥 Momentum（強化）
# =========================
df["Momentum_20"] = (
    0.5 * df["Return_20"]
    + 0.3 * df["Return_10"]
    + 0.2 * df["Return_5"]
)

df["Momentum_accel"] = df["Return_20"] - df["Return_10"]

# =========================
# 🔥 移動平均
# =========================
for w in [5, 10, 20, 30]:
    ma = df.groupby("Ticker")["Close"].transform(
        lambda x: x.shift(1).rolling(w).mean()
    )
    df[f"MA{w}_ratio"] = df["Close"].shift(1) / (ma + 1e-9)

# =========================
# 🔥 ボラ
# =========================
df["Volatility"] = df.groupby("Ticker")["Close"].pct_change().shift(1).transform(
    lambda x: x.rolling(20).std()
)

# =========================
# 🔥 トレンド
# =========================
df["Trend_10"] = df.groupby("Ticker")["Close"].shift(1) / df.groupby("Ticker")["Close"].shift(11) - 1
df["Trend_20"] = df.groupby("Ticker")["Close"].shift(1) / df.groupby("Ticker")["Close"].shift(21) - 1
df["Trend_40"] = df.groupby("Ticker")["Close"].shift(1) / df.groupby("Ticker")["Close"].shift(41) - 1

def zscore(x):
    return (x - x.rolling(Z_WINDOW).mean()) / (x.rolling(Z_WINDOW).std() + 1e-9)

df["Trend_10_z"] = df.groupby("Ticker")["Trend_10"].transform(zscore)
df["Trend_20_z"] = df.groupby("Ticker")["Trend_20"].transform(zscore)
df["Trend_40_z"] = df.groupby("Ticker")["Trend_40"].transform(zscore)

# =========================
# 🔥 ドローダウン
# =========================
rolling_max_20 = df.groupby("Ticker")["Close"].transform(
    lambda x: x.shift(1).rolling(20).max()
)
rolling_max_40 = df.groupby("Ticker")["Close"].transform(
    lambda x: x.shift(1).rolling(40).max()
)

df["DD_20"] = df["Close"].shift(1) / (rolling_max_20 + 1e-9) - 1
df["DD_40"] = df["Close"].shift(1) / (rolling_max_40 + 1e-9) - 1

# =========================
# 🔥 Trend効率（修正済）
# =========================
df["TrendVol"] = df["Trend_20"] / (df["Volatility"] * np.sqrt(20) + 1e-6)

# =========================
# 🔥 出来高
# =========================
vol_mean = df.groupby("Ticker")["Volume"].transform(
    lambda x: x.shift(1).rolling(20).mean()
)
vol_std = df.groupby("Ticker")["Volume"].transform(
    lambda x: x.shift(1).rolling(20).std()
)

df["Volume_Z"] = (df["Volume"].shift(1) - vol_mean) / (vol_std + 1e-9)

# =========================
# 🔥 市場特徴量（強化）
# =========================
market_mean = df.groupby("Date")["Return_5"].transform("mean")
market_std  = df.groupby("Date")["Return_5"].transform("std")

df["Market_Z"] = (df["Return_5"] - market_mean) / (market_std + 1e-9)
df["Market_Trend"] = df.groupby("Date")["Trend_20"].transform("mean")

df["Market_Vol"] = market_std
df["Market_Trend_Str"] = df.groupby("Date")["Trend_20"].transform(lambda x: x.abs().mean())

# 🔥 追加（超重要）
df["Market_Sharpe"] = market_mean / (market_std + 1e-9)

# =========================
# 🔥 クロスセクション
# =========================
rank_cols = [
    "Return_10",
    "Trend_20_z",
    "TrendVol",
    "DD_20"
]

for col in rank_cols:
    df[f"{col}_rank"] = df.groupby("Date")[col].rank(pct=True)

# =========================
# 🎯 Target（現実型）
# =========================
future_close = df.groupby("Ticker")["Close"].shift(-HOLD_DAYS)
future_max = df.groupby("Ticker")["High"].shift(-1).rolling(HOLD_DAYS).max()

df["Target"] = (
    0.5 * (future_close / df["Close"] - 1)
    + 0.5 * (future_max / df["Close"] - 1)
)

df["Target"] = df["Target"].clip(-TARGET_CLIP, TARGET_CLIP)

# =========================
# FEATURES
# =========================
FEATURES = [
    "Return_5","Return_10","Return_20",
    "Momentum_20","Momentum_accel",
    "MA5_ratio","MA10_ratio","MA20_ratio","MA30_ratio",
    "Volatility",
    "Trend_10_z","Trend_20_z","Trend_40_z",
    "DD_20","DD_40",
    "TrendVol","Volume_Z",
    "Return_10_rank","Trend_20_z_rank",
    "TrendVol_rank","DD_20_rank",
    "Market_Z","Market_Trend",
    "Market_Vol","Market_Trend_Str","Market_Sharpe"
]

# =========================
# データ作成
# =========================
train_df = df.dropna(subset=FEATURES + ["Target"]).reset_index(drop=True)

latest_date = df["Date"].max()
predict_df = df[df["Date"] == latest_date].dropna(subset=FEATURES).reset_index(drop=True)

# =========================
# 保存
# =========================
train_df.to_parquet(TRAIN_SAVE_PATH)
predict_df.to_parquet(PREDICT_SAVE_PATH)

print("\n保存完了（7日モデル・完成版）")
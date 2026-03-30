import pandas as pd
import numpy as np
import duckdb
import os

# =========================
# 設定
# =========================
PARQUET_FILE = "stock_data/prices.parquet"

TRAIN_SAVE_PATH = "ml_dataset.parquet"
PREDICT_SAVE_PATH = "ml_dataset_latest.parquet"

EARNINGS_FILE = "stock_data/earnings.parquet"

HOLD_DAYS = 3

# =========================
# データ読み込み
# =========================
con = duckdb.connect()

df = con.execute(f"""
SELECT 
    Date,
    Ticker,
    Name,
    Open,
    High,
    Low,
    Close,
    Volume
FROM '{PARQUET_FILE}'
ORDER BY Ticker, Date
""").df()

print("元データサイズ:", df.shape)

df["Date"] = pd.to_datetime(df["Date"])


# =========================
# 🔥 基本特徴量
# =========================
df["Return_1"] = df.groupby("Ticker")["Close"].pct_change()
df["Return_3"] = df.groupby("Ticker")["Close"].pct_change(3)

df["MA3"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(3).mean())
df["MA5"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(5).mean())
df["MA10"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(10).mean())

df["MA3_ratio"] = df["Close"] / df["MA3"]
df["MA5_ratio"] = df["Close"] / df["MA5"]
df["MA10_ratio"] = df["Close"] / df["MA10"]

df["Volatility"] = (
    df.groupby("Ticker")["Return_1"]
    .transform(lambda x: x.rolling(10).std())
)

df["Volume_change"] = df.groupby("Ticker")["Volume"].pct_change()
df["Volume_ma5"] = df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(5).mean())
df["Volume_ratio"] = df["Volume"] / df["Volume_ma5"]

df["HL_range"] = (df["High"] - df["Low"]) / df["Close"]


# =========================
# RSI
# =========================
delta = df.groupby("Ticker")["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.groupby(df["Ticker"]).transform(lambda x: x.rolling(7).mean())
avg_loss = loss.groupby(df["Ticker"]).transform(lambda x: x.rolling(7).mean())

rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))


# =========================
# 🔥 ストップ高検出
# =========================
df["limit_up_raw"] = (df["Return_1"] > 0.15).astype(int)
df["limit_up_flag"] = df.groupby("Ticker")["limit_up_raw"].shift(1).fillna(0)


# =========================================================
# 🔥 決算（外部キャッシュ版・重要）
# =========================================================
if os.path.exists(EARNINGS_FILE):
    earnings_df = pd.read_parquet(EARNINGS_FILE)
    print("⚡ earnings: キャッシュ使用")
else:
    print("⚠ earningsファイルなし（先に取得処理を実行）")
    earnings_df = pd.DataFrame(columns=["Ticker", "Date"])


df["is_earnings"] = 0

if not earnings_df.empty:

    earnings_df["Date"] = pd.to_datetime(earnings_df["Date"]).dt.date

    earnings_map = earnings_df.groupby("Ticker")["Date"].apply(list).to_dict()

    for ticker, dates in earnings_map.items():

        ticker_mask = df["Ticker"] == ticker

        for d in dates:
            mask = ticker_mask & (
                (df["Date"].dt.date >= d - pd.Timedelta(days=1)) &
                (df["Date"].dt.date <= d + pd.Timedelta(days=1))
            )
            df.loc[mask, "is_earnings"] = 1


# =========================
# クロスセクションランキング
# =========================
rank_features = [
    "Return_1",
    "Return_3",
    "MA3_ratio",
    "MA5_ratio",
    "MA10_ratio",
    "Volatility",
    "Volume_change",
    "Volume_ratio",
    "HL_range",
    "RSI"
]

for col in rank_features:
    df[col + "_rank"] = df.groupby("Date")[col].rank(pct=True)


# =========================
# Target（3日先リターン）
# =========================
df["FutureReturn_3"] = (
    df.groupby("Ticker")["Close"].shift(-HOLD_DAYS) / df["Close"] - 1
)

df["Target"] = df.groupby("Date")["FutureReturn_3"].rank(pct=True)


# =========================
# 無限値処理
# =========================
df = df.replace([np.inf, -np.inf], np.nan)


# =========================
# 学習データ
# =========================
train_df = df.dropna(subset=[
    "Return_1_rank",
    "Return_3_rank",
    "MA3_ratio_rank",
    "MA5_ratio_rank",
    "MA10_ratio_rank",
    "Volatility_rank",
    "Volume_change_rank",
    "Volume_ratio_rank",
    "HL_range_rank",
    "RSI_rank",
    "FutureReturn_3"
]).copy()

train_df = train_df.reset_index(drop=True)


# =========================
# 予測データ
# =========================
predict_df = df.dropna(subset=[
    "Return_1_rank",
    "Return_3_rank",
    "MA3_ratio_rank",
    "MA5_ratio_rank",
    "MA10_ratio_rank",
    "Volatility_rank",
    "Volume_change_rank",
    "Volume_ratio_rank",
    "HL_range_rank",
    "RSI_rank"
]).copy()

predict_df = predict_df.reset_index(drop=True)


# =========================
# デバッグ
# =========================
print("\n=== TRAIN DEBUG ===")
print("rows:", len(train_df))
print("latest:", train_df["Date"].max())

print("\n=== PREDICT DEBUG ===")
print("rows:", len(predict_df))
print("latest:", predict_df["Date"].max())


# =========================
# 保存
# =========================
train_df.to_parquet(TRAIN_SAVE_PATH)
predict_df.to_parquet(PREDICT_SAVE_PATH)

print("\n保存完了")
print("train rows:", len(train_df))
print("predict rows:", len(predict_df))
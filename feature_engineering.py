import pandas as pd
import numpy as np
import duckdb
import yfinance as yf
import os
import pickle

# =========================
# 設定
# =========================
PARQUET_FILE = "stock_data/prices.parquet"

TRAIN_SAVE_PATH = "ml_dataset.parquet"
PREDICT_SAVE_PATH = "ml_dataset_latest.parquet"

HOLD_DAYS = 3

# キャッシュ（超重要：高速化）
CACHE_FILE = "earnings_cache.pkl"

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
# 🔥 短期特徴量
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
# RSI（短期）
# =========================
delta = df.groupby("Ticker")["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.groupby(df["Ticker"]).transform(lambda x: x.rolling(7).mean())
avg_loss = loss.groupby(df["Ticker"]).transform(lambda x: x.rolling(7).mean())

rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

# =========================
# 🔥 ストップ高検出（超重要）
# =========================
df["limit_up_raw"] = (df["Return_1"] > 0.15).astype(int)

# 翌日フラグに変換
df["limit_up_flag"] = df.groupby("Ticker")["limit_up_raw"].shift(1)
df["limit_up_flag"] = df["limit_up_flag"].fillna(0)

# =========================
# 🔥 決算フラグ（キャッシュ付き）
# =========================
def get_earnings_dates(ticker):
    try:
        t = yf.Ticker(ticker)
        df_e = t.earnings_dates

        if df_e is None or len(df_e) == 0:
            return []

        return pd.to_datetime(df_e.index).date.tolist()
    except:
        return []

# キャッシュ読み込み
if os.path.exists(CACHE_FILE):
    earnings_cache = pickle.load(open(CACHE_FILE, "rb"))
else:
    earnings_cache = {}

df["is_earnings"] = 0

tickers = df["Ticker"].unique()

for i, ticker in enumerate(tickers):

    if ticker in earnings_cache:
        dates = earnings_cache[ticker]
    else:
        print(f"決算取得中 {i+1}/{len(tickers)}: {ticker}")
        dates = get_earnings_dates(ticker)
        earnings_cache[ticker] = dates

    if len(dates) == 0:
        continue

    ticker_mask = df["Ticker"] == ticker

    for d in dates:
        mask = ticker_mask & (
            (df["Date"].dt.date >= d - pd.Timedelta(days=1)) &
            (df["Date"].dt.date <= d + pd.Timedelta(days=1))
        )
        df.loc[mask, "is_earnings"] = 1

# キャッシュ保存
pickle.dump(earnings_cache, open(CACHE_FILE, "wb"))

# =========================
# クロスセクション特徴量
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
# 🔥 Target（3日）
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
# 学習用
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
# 予測用
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
print("\n=== TRAIN DATA DEBUG ===")
print("rows:", len(train_df))
print("latest:", train_df["Date"].max())
print("========================")

print("\n=== PREDICT DATA DEBUG ===")
print("rows:", len(predict_df))
print("latest:", predict_df["Date"].max())
print("========================")

# =========================
# 保存
# =========================
train_df.to_parquet(TRAIN_SAVE_PATH)
predict_df.to_parquet(PREDICT_SAVE_PATH)

print("\n保存完了")
print("train rows:", len(train_df))
print("predict rows:", len(predict_df))
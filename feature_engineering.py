import pandas as pd
import numpy as np
import duckdb

# =========================
# 設定
# =========================
PARQUET_FILE = "stock_data/prices.parquet"

TRAIN_SAVE_PATH = "ml_dataset.parquet"
PREDICT_SAVE_PATH = "ml_dataset_latest.parquet"

HOLD_DAYS = 5

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

# =========================
# 型整備
# =========================
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

# =========================
# 基本特徴量
# =========================
df["Return_1"] = df.groupby("Ticker")["Close"].pct_change()

# 🔥 モメンタム（最重要）
df["Return_5"] = df.groupby("Ticker")["Close"].pct_change(5)
df["Return_10"] = df.groupby("Ticker")["Close"].pct_change(10)
df["Return_20"] = df.groupby("Ticker")["Close"].pct_change(20)

# =========================
# 移動平均
# =========================
df["MA5"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(5).mean())
df["MA25"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(25).mean())
df["MA75"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(75).mean())

df["MA5_ratio"] = df["Close"] / df["MA5"]
df["MA25_ratio"] = df["Close"] / df["MA25"]
df["MA75_ratio"] = df["Close"] / df["MA75"]

# =========================
# ボラ・出来高
# =========================
df["Volatility"] = df.groupby("Ticker")["Return_1"].transform(
    lambda x: x.rolling(20).std()
)

df["Volume_change"] = df.groupby("Ticker")["Volume"].pct_change()

# 🔥 出来高スパイク
df["Volume_MA20"] = df.groupby("Ticker")["Volume"].transform(
    lambda x: x.rolling(20).mean()
)
df["Volume_spike"] = df["Volume"] / df["Volume_MA20"]

# =========================
# 値動き強さ
# =========================
df["HL_range"] = (df["High"] - df["Low"]) / df["Close"]

# 🔥 ブレイクアウト
df["High_20"] = df.groupby("Ticker")["High"].transform(
    lambda x: x.rolling(20).max()
)
df["Breakout"] = df["Close"] / df["High_20"]

# 🔥 ボラ調整リターン
df["Return_vol_adj"] = df["Return_5"] / (df["Volatility"] + 1e-9)

# =========================
# RSI
# =========================
delta = df.groupby("Ticker")["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.groupby(df["Ticker"]).transform(lambda x: x.rolling(14).mean())
avg_loss = loss.groupby(df["Ticker"]).transform(lambda x: x.rolling(14).mean())

rs = avg_gain / (avg_loss + 1e-9)
df["RSI"] = 100 - (100 / (1 + rs))

# =========================
# 🔥 クロスセクションrank
# =========================
rank_features = [
    "Return_1",
    "MA5_ratio",
    "MA25_ratio",
    "MA75_ratio",
    "Volatility",
    "Volume_change",
    "HL_range",
    "RSI",
    "Return_5",
    "Return_20",
    "Volume_spike",
    "Breakout",
    "Return_vol_adj"
]

for col in rank_features:
    df[col + "_rank"] = df.groupby("Date")[col].rank(pct=True)

# =========================
# 🔥 Target（改良版）
# =========================
df["FutureReturn_5"] = (
    df.groupby("Ticker")["Close"].shift(-HOLD_DAYS) / df["Close"] - 1
)

# 👉 2種類用意（あとで選べる）

df["Target_rank"] = df.groupby("Date")["FutureReturn_5"].transform(
    lambda x: x.rank(pct=True)
)

# =========================
# 無限値処理
# =========================
df = df.replace([np.inf, -np.inf], np.nan)

# =========================
# 🔥 学習データ
# =========================
feature_cols = [c + "_rank" for c in rank_features] + [
    "Return_5",
    "Return_20",
    "Breakout",
    "Volume_spike",
    "Return_vol_adj"
]

train_df = df.dropna(subset=feature_cols + ["Target_rank"]).copy()

train_df = train_df[
    train_df["FutureReturn_5"].notna()
].copy()

train_df = train_df.reset_index(drop=True)

# =========================
# 🔥 予測データ
# =========================
predict_df = df.dropna(subset=feature_cols).copy()

predict_df = predict_df.drop(columns=[
    "FutureReturn_5",
    "Target_rank",
    "Target_raw"
], errors="ignore")

predict_df = predict_df.reset_index(drop=True)

# =========================
# デバッグ
# =========================
print("\n=== TRAIN DATA DEBUG ===")
print("rows:", len(train_df))
print("latest:", train_df["Date"].max())
print("unique dates:", train_df["Date"].nunique())

print("\n=== PREDICT DATA DEBUG ===")
print("rows:", len(predict_df))
print("latest:", predict_df["Date"].max())
print("unique dates:", predict_df["Date"].nunique())

# =========================
# 保存
# =========================
train_df.to_parquet(TRAIN_SAVE_PATH)
predict_df.to_parquet(PREDICT_SAVE_PATH)

print("\n保存完了")
print("train rows:", len(train_df))
print("predict rows:", len(predict_df))
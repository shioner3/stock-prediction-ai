import pandas as pd
import numpy as np
import duckdb

# =========================
# 設定
# =========================
PARQUET_FILE = "stock_data/prices.parquet"

TRAIN_SAVE_PATH = "ml_dataset.parquet"
PREDICT_SAVE_PATH = "ml_dataset_latest.parquet"

HOLD_DAYS = 3
MIN_COUNT = 3000
Z_WINDOW = 20

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
""").df()

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# =========================
# 日付フィルター
# =========================
counts = df["Date"].value_counts()
valid_dates = counts[counts >= MIN_COUNT].index
df = df[df["Date"].isin(valid_dates)].copy()

# =========================
# 🔥 基本特徴量（過去のみ）
# =========================

# リターン
df["Return_1"] = df.groupby("Ticker")["Close"].pct_change().shift(1)
df["Return_3"] = df.groupby("Ticker")["Close"].pct_change(3).shift(1)

# MA
for w in [3, 5, 10]:
    df[f"MA{w}"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(w).mean()
    ).shift(1)

    df[f"MA{w}_ratio"] = df["Close"].shift(1) / df[f"MA{w}"]

# ボラ
df["Volatility"] = df.groupby("Ticker")["Return_1"].transform(
    lambda x: x.rolling(10).std()
).shift(1)

# 出来高
df["Volume_change"] = df.groupby("Ticker")["Volume"].pct_change().shift(1)

df["Volume_ma5"] = df.groupby("Ticker")["Volume"].transform(
    lambda x: x.rolling(5).mean()
).shift(1)

df["Volume_ratio"] = df["Volume"].shift(1) / df["Volume_ma5"]

# 高値安値
df["HL_range"] = ((df["High"] - df["Low"]) / df["Close"]).shift(1)

# =========================
# 市場（リーク防止済）
# =========================
df["Market_Return_1"] = df.groupby("Date")["Return_1"].transform("mean").shift(1)
df["Rel_Return_1"] = df["Return_1"] - df["Market_Return_1"]

# =========================
# トレンド
# =========================
df["Trend_5"] = df["Close"].shift(1) / df.groupby("Ticker")["Close"].shift(6) - 1
df["Trend_10"] = df["Close"].shift(1) / df.groupby("Ticker")["Close"].shift(11) - 1

# =========================
# Zスコア
# =========================
trend5_mean = df.groupby("Ticker")["Trend_5"].transform(lambda x: x.rolling(Z_WINDOW).mean())
trend5_std  = df.groupby("Ticker")["Trend_5"].transform(lambda x: x.rolling(Z_WINDOW).std())
df["Trend_5_z"] = (df["Trend_5"] - trend5_mean) / (trend5_std + 1e-9)

trend10_mean = df.groupby("Ticker")["Trend_10"].transform(lambda x: x.rolling(Z_WINDOW).mean())
trend10_std  = df.groupby("Ticker")["Trend_10"].transform(lambda x: x.rolling(Z_WINDOW).std())
df["Trend_10_z"] = (df["Trend_10"] - trend10_mean) / (trend10_std + 1e-9)

# =========================
# 🔥 追加特徴量（完全リーク対策版）
# =========================

# ① ギャップ（前日終値ベース）
df["Gap"] = df["Open"].shift(1) / df["Close"].shift(2) - 1

# ② ボラ変化（過去ボラのみ）
df["Volatility_change"] = df["Volatility"] / df.groupby("Ticker")["Volatility"].shift(5)

# ③ 出来高スパイク（過去のみ）
vol_mean_20 = df.groupby("Ticker")["Volume"].transform(
    lambda x: x.rolling(20).mean()
).shift(1)

df["Volume_spike"] = df["Volume"].shift(1) / vol_mean_20

# ④ モメンタム加速（すでに過去化されてるのでOK）
df["Momentum_acc"] = df["Return_1"] - df["Return_3"]

# =========================
# 🎯 ターゲット（生値）
# =========================
df["FutureReturn"] = (
    df.groupby("Ticker")["Close"].shift(-HOLD_DAYS) / df["Close"] - 1
)

# 外れ値カット
df["FutureReturn"] = df["FutureReturn"].clip(-0.2, 0.2)

df["Target"] = df["FutureReturn"]

# 欠損削除
df = df.dropna(subset=["Target"])

# =========================
# FEATURES
# =========================
FEATURES = [
    "Return_1","Return_3",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio",
    "HL_range",
    "Rel_Return_1",
    "Trend_5_z","Trend_10_z",
    
    # 🔥 追加分
    "Gap",
    "Volatility_change",
    "Volume_spike",
    "Momentum_acc"
]

# =========================
# データ作成
# =========================
train_df = df.dropna(subset=FEATURES + ["Target"]).reset_index(drop=True)

latest_date = df["Date"].max()
predict_df = df[df["Date"] == latest_date].dropna(subset=FEATURES).reset_index(drop=True)

# =========================
# デバッグ
# =========================
print("\n=== TARGET CHECK（生値）===")
print(train_df["Target"].describe())

print("\n=== DATE CHECK ===")
print("Train:", train_df["Date"].min(), "→", train_df["Date"].max())
print("Predict:", latest_date)

# =========================
# 保存
# =========================
train_df.to_parquet(TRAIN_SAVE_PATH)
predict_df.to_parquet(PREDICT_SAVE_PATH)

print("\n保存完了（追加特徴量＋リーク対策済み）")
import pandas as pd
import numpy as np
import duckdb
import os

# =========================
# 設定
# =========================
PARQUET_FILE = "stock_data/prices.parquet"
SAVE_PATH = "stock_data/technical_features.parquet"

MIN_HISTORY = 100

# =========================
# データ読み込み
# =========================
print("Loading parquet...")

df = duckdb.query(f"""
    SELECT *
    FROM read_parquet('{PARQUET_FILE}')
""").to_df()

# =========================
# 前処理
# =========================
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# 数値化
cols = ["Open", "High", "Low", "Close", "Volume"]

for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# =========================
# 特徴量生成関数
# =========================
def create_features(g):

    g = g.copy()

    # =========================
    # リターン系
    # =========================
    g["Return_1"] = g["Close"].pct_change(1)
    g["Return_3"] = g["Close"].pct_change(3)
    g["Return_5"] = g["Close"].pct_change(5)
    g["Return_10"] = g["Close"].pct_change(10)
    g["Return_20"] = g["Close"].pct_change(20)

    # =========================
    # 移動平均
    # =========================
    for w in [5, 10, 20, 25, 50, 75]:

        g[f"SMA_{w}"] = g["Close"].rolling(w).mean()
        g[f"EMA_{w}"] = g["Close"].ewm(span=w, adjust=False).mean()

        # 乖離率
        g[f"SMA_Gap_{w}"] = (
            g["Close"] / g[f"SMA_{w}"] - 1
        )

        g[f"EMA_Gap_{w}"] = (
            g["Close"] / g[f"EMA_{w}"] - 1
        )

    # =========================
    # 高値・安値更新
    # =========================
    for w in [5, 10, 20, 60]:

        g[f"High_{w}"] = g["High"].rolling(w).max()
        g[f"Low_{w}"] = g["Low"].rolling(w).min()

        g[f"Breakout_High_{w}"] = (
            g["Close"] > g[f"High_{w}"].shift(1)
        ).astype(int)

        g[f"Breakout_Low_{w}"] = (
            g["Close"] < g[f"Low_{w}"].shift(1)
        ).astype(int)

    # =========================
    # ボラティリティ
    # =========================
    for w in [5, 10, 20]:

        g[f"Volatility_{w}"] = (
            g["Return_1"].rolling(w).std()
        )

    # =========================
    # RSI
    # =========================
    delta = g["Close"].diff()

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain = pd.Series(gain, index=g.index)
    loss = pd.Series(loss, index=g.index)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / (avg_loss + 1e-9)

    g["RSI_14"] = 100 - (100 / (1 + rs))

    # =========================
    # MACD
    # =========================
    ema12 = g["Close"].ewm(span=12, adjust=False).mean()
    ema26 = g["Close"].ewm(span=26, adjust=False).mean()

    g["MACD"] = ema12 - ema26
    g["MACD_Signal"] = g["MACD"].ewm(span=9, adjust=False).mean()
    g["MACD_Hist"] = g["MACD"] - g["MACD_Signal"]

    # =========================
    # ボリンジャーバンド
    # =========================
    ma20 = g["Close"].rolling(20).mean()
    std20 = g["Close"].rolling(20).std()

    g["BB_Upper"] = ma20 + std20 * 2
    g["BB_Lower"] = ma20 - std20 * 2

    g["BB_Position"] = (
        (g["Close"] - g["BB_Lower"]) /
        (g["BB_Upper"] - g["BB_Lower"] + 1e-9)
    )

    # =========================
    # 出来高
    # =========================
    for w in [5, 20]:

        g[f"Volume_MA_{w}"] = g["Volume"].rolling(w).mean()

        g[f"Volume_Ratio_{w}"] = (
            g["Volume"] / (g[f"Volume_MA_{w}"] + 1e-9)
        )

    # =========================
    # ローソク足
    # =========================
    g["Body"] = (
        g["Close"] - g["Open"]
    )

    g["Body_Ratio"] = (
        (g["Close"] - g["Open"]) /
        (g["High"] - g["Low"] + 1e-9)
    )

    g["Upper_Shadow"] = (
        g["High"] - g[["Open", "Close"]].max(axis=1)
    )

    g["Lower_Shadow"] = (
        g[["Open", "Close"]].min(axis=1) - g["Low"]
    )

    # =========================
    # ギャップ
    # =========================
    g["Gap"] = (
        g["Open"] / g["Close"].shift(1) - 1
    )

    # =========================
    # トレンド判定
    # =========================
    g["Trend_Up"] = (
        (g["SMA_5"] > g["SMA_25"]) &
        (g["SMA_25"] > g["SMA_75"])
    ).astype(int)

    g["Trend_Down"] = (
        (g["SMA_5"] < g["SMA_25"]) &
        (g["SMA_25"] < g["SMA_75"])
    ).astype(int)

    # =========================
    # GC / DC
    # =========================
    g["Golden_Cross"] = (
        (g["SMA_5"] > g["SMA_25"]) &
        (g["SMA_5"].shift(1) <= g["SMA_25"].shift(1))
    ).astype(int)

    g["Dead_Cross"] = (
        (g["SMA_5"] < g["SMA_25"]) &
        (g["SMA_5"].shift(1) >= g["SMA_25"].shift(1))
    ).astype(int)

    # =========================
    # 連騰・連落
    # =========================
    up = (g["Close"] > g["Close"].shift(1)).astype(int)
    down = (g["Close"] < g["Close"].shift(1)).astype(int)

    g["Up_Count_5"] = up.rolling(5).sum()
    g["Down_Count_5"] = down.rolling(5).sum()

    # =========================
    # ATR
    # =========================
    tr1 = g["High"] - g["Low"]
    tr2 = abs(g["High"] - g["Close"].shift(1))
    tr3 = abs(g["Low"] - g["Close"].shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    g["ATR_14"] = tr.rolling(14).mean()

    # =========================
    # squeeze系
    # =========================
    bb_width = (
        (g["BB_Upper"] - g["BB_Lower"]) /
        (ma20 + 1e-9)
    )

    g["BB_Width"] = bb_width

    g["Squeeze"] = (
        bb_width <
        bb_width.rolling(50).quantile(0.2)
    ).astype(int)

    return g


# =========================
# 銘柄ごと処理
# =========================
print("Creating features...")

df = (
    df.groupby("Ticker", group_keys=False)
    .filter(lambda x: len(x) >= MIN_HISTORY)
)

df_feat = (
    df.groupby("Ticker", group_keys=False)
    .apply(create_features)
)

# =========================
# NaN整理
# =========================
df_feat = df_feat.replace([np.inf, -np.inf], np.nan)

# =========================
# 保存
# =========================
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

df_feat.to_parquet(SAVE_PATH, index=False)

# =========================
# 完了
# =========================
print("\n=== DONE ===")
print(df_feat.shape)

print("\nColumns:")
print(df_feat.columns.tolist())

print("\nSaved:")
print(SAVE_PATH)
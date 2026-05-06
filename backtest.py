import pandas as pd
import numpy as np

from signal_engine import generate_signals

# =========================
# 設定
# =========================
FEATURE_PATH = "stock_data/features.parquet"

MAX_HOLD_DAYS = 15
MAX_POSITIONS = 1
INITIAL_CAPITAL = 1.0
COST = 0.001

# =========================
# Exitロジック
# =========================
def get_exit_date(df_ticker, entry_idx, max_hold=15):

    entry_price = df_ticker.iloc[entry_idx + 1]["Open"]

    for i in range(1, max_hold + 1):

        if entry_idx + i >= len(df_ticker):
            break

        row = df_ticker.iloc[entry_idx + i]

        ret = row["Close"] / entry_price - 1

        if ret > 0.15:
            return row["Date"]

        if (
            (row["return_3d"] < -0.02) or
            (row["ma5_diff"] < -0.03) or
            (row["market_trend_5"] < -0.002)
        ):
            return row["Date"]

    return df_ticker.iloc[min(entry_idx + max_hold, len(df_ticker)-1)]["Date"]

# =========================
# データ
# =========================
df = pd.read_parquet(FEATURE_PATH)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"])

df = generate_signals(df)

print("Total rows:", len(df))
print("Signal count:", df["signal"].sum())

# =========================
# シグナル
# =========================
trades = df[df["signal"]].copy()

trades["rank"] = trades.groupby("Date")["signal_score"] \
    .rank(ascending=False, method="first")

trades = trades[trades["rank"] <= MAX_POSITIONS]

print("Trades:", len(trades))

# =========================
# ポジション生成（CLEAN）
# =========================
grouped = df.groupby("Ticker")

positions = []

for _, row in trades.iterrows():

    ticker = row["Ticker"]
    entry_date = row["Date"]

    df_t = grouped.get_group(ticker).reset_index(drop=True)

    idx = df_t.index[df_t["Date"] == entry_date]
    if len(idx) == 0:
        continue

    entry_idx = idx[0]

    if entry_idx + 1 >= len(df_t):
        continue

    exit_date = get_exit_date(df_t, entry_idx, MAX_HOLD_DAYS)

    entry_price = df_t.iloc[entry_idx + 1]["Open"]

    exit_idx = df_t.index[df_t["Date"] == exit_date]
    if len(exit_idx) == 0:
        continue

    exit_idx = exit_idx[0]
    exit_price = df_t.iloc[exit_idx]["Close"]

    # =========================
    # ★単純リターン（これが正解）
    # =========================
    ret = exit_price / entry_price - 1 - COST * 2

    positions.append({
        "entry_date": entry_date,
        "exit_date": exit_date,
        "ret": ret,
        "ticker": ticker
    })

pos_df = pd.DataFrame(positions)

if len(pos_df) == 0:
    print("❌ No positions")
    exit()

# =========================
# 日次シミュレーション（正規版）
# =========================
dates = sorted(df["Date"].unique())

equity = []
capital = INITIAL_CAPITAL

for date in dates:

    active = pos_df[
        (pos_df["entry_date"] <= date) &
        (pos_df["exit_date"] > date)
    ]

    if len(active) == 0:
        equity.append(capital)
        continue

    # =========================
    # ★ここが本質（平均リターン）
    # =========================
    daily_ret = active["ret"].mean()

    capital *= (1 + daily_ret)

    equity.append(capital)

equity = pd.Series(equity, index=dates)

# =========================
# 指標
# =========================
rets = equity.pct_change().dropna()

cagr = equity.iloc[-1] ** (252 / len(equity)) - 1

sharpe = rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)

max_dd = (equity / equity.cummax() - 1).min()

# =========================
# 出力
# =========================
print("\n=== CLEAN BACKTEST RESULT ===")
print(f"CAGR  : {cagr:.4f}")
print(f"Sharpe: {sharpe:.4f}")
print(f"MaxDD : {max_dd:.4f}")
print(f"Trades: {len(pos_df)}")

equity.to_csv("equity_curve.csv")
pos_df.to_csv("trades.csv", index=False)
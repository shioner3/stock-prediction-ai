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
# Exit関数（改良済み）
# =========================
def get_exit_date(df_ticker, entry_idx, max_hold=15):

    entry_price = df_ticker.iloc[entry_idx + 1]["Open"]

    for i in range(1, max_hold + 1):

        if entry_idx + i >= len(df_ticker):
            break

        row = df_ticker.iloc[entry_idx + i]

        current_price = row["Close"]
        current_ret = current_price / entry_price - 1

        # 利確
        if current_ret > 0.15:
            return row["Date"]

        # 損切り（やや緩め）
        if (
            (row["return_3d"] < -0.02) or
            (row["ma5_diff"] < -0.03) or
            (row["market_trend_5"] < -0.002)
        ):
            return row["Date"]

    return df_ticker.iloc[min(entry_idx + max_hold, len(df_ticker)-1)]["Date"]

# =========================
# 読み込み
# =========================
df = pd.read_parquet(FEATURE_PATH)

if len(df) == 0:
    print("❌ features is empty")
    exit()

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

# =========================
# シグナル
# =========================
df = generate_signals(df)

if "signal" not in df.columns:
    print("❌ signal column missing")
    exit()

print("Total rows:", len(df))
print("Signal count:", df["signal"].sum())

# =========================
# トレード抽出
# =========================
trades = df[df["signal"]].copy()

if len(trades) == 0:
    print("❌ No trades generated")
    exit()

# =========================
# A: 同時ポジション制御（強制1）
# =========================
trades["rank"] = (
    trades.groupby("Date")["signal_score"]
    .rank(ascending=False, method="first")
)

trades = trades[trades["rank"] <= MAX_POSITIONS]

print("Trades after rank filter:", len(trades))

if len(trades) == 0:
    print("❌ No trades after ranking")
    exit()

# =========================
# B: 保有中重複エントリー禁止
# =========================
grouped = df.groupby("Ticker")

last_exit = {}

positions = []

# =========================
# ポジション生成
# =========================
for _, row in trades.iterrows():

    ticker = row["Ticker"]
    entry_date = row["Date"]

    # ★保有中スキップ
    if ticker in last_exit:
        if entry_date <= last_exit[ticker]:
            continue

    try:
        df_t = grouped.get_group(ticker).reset_index(drop=True)
    except:
        continue

    idx_list = df_t.index[df_t["Date"] == entry_date]

    if len(idx_list) == 0:
        continue

    entry_idx = idx_list[0]

    if entry_idx + 1 >= len(df_t):
        continue

    # ===== exit =====
    exit_date = get_exit_date(df_t, entry_idx, MAX_HOLD_DAYS)

    try:
        entry_price = df_t.iloc[entry_idx + 1]["Open"]
        exit_price = df_t[df_t["Date"] == exit_date]["Close"].values[0]
    except:
        continue

    ret = exit_price / entry_price - 1 - COST * 2

    hold_days = (exit_date - entry_date).days
    if hold_days <= 0:
        continue

    daily_ret = (1 + ret) ** (1 / hold_days) - 1

    positions.append({
        "entry_date": entry_date,
        "exit_date": exit_date,
        "daily_ret": daily_ret,
        "ticker": ticker
    })

    # ★更新（C）
    last_exit[ticker] = exit_date

pos_df = pd.DataFrame(positions)

if len(pos_df) == 0:
    print("❌ No positions")
    exit()

# =========================
# C: 日次シミュレーション（正規化）
# =========================
dates = sorted(df["Date"].unique())

capital = INITIAL_CAPITAL
equity_curve = []

for date in dates:

    active = pos_df[
        (pos_df["entry_date"] <= date) &
        (pos_df["exit_date"] > date)
    ]

    if len(active) == 0:
        equity_curve.append(capital)
        continue

    # ★ 重要修正：過剰レバ防止
    daily_ret = active["daily_ret"].mean()

    capital *= (1 + daily_ret)

    equity_curve.append(capital)

equity = pd.Series(equity_curve, index=dates)

if len(equity) == 0:
    print("❌ equity empty")
    exit()

# =========================
# 指標
# =========================
daily_returns = equity.pct_change().dropna()

if len(daily_returns) == 0:
    print("❌ no daily returns")
    exit()

cagr = equity.iloc[-1] ** (252 / len(equity)) - 1

sharpe = (
    daily_returns.mean() /
    (daily_returns.std() + 1e-9)
) * np.sqrt(252)

max_dd = (
    (equity / equity.cummax()) - 1
).min()

# =========================
# 出力
# =========================
print("\n=== RESULT ===")
print(f"CAGR  : {cagr:.4f}")
print(f"Sharpe: {sharpe:.4f}")
print(f"MaxDD : {max_dd:.4f}")
print(f"Trades: {len(trades)}")

# =========================
# 保存
# =========================
equity.to_csv("equity_curve.csv")
pos_df.to_csv("trades.csv", index=False)
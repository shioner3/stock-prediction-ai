import pandas as pd
import numpy as np

# =========================
# 設定
# =========================
FEATURE_PATH = "stock_data/features.parquet"

HOLD_DAYS = 5
MAX_POSITIONS = 3
INITIAL_CAPITAL = 1.0
COST = 0.001  # 0.1%

# =========================
# 読み込み
# =========================
df = pd.read_parquet(FEATURE_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(["Date", "Ticker"])

# =========================
# シグナル読み込み
# =========================
from signal_engine import generate_signals

df = generate_signals(df)

# =========================
# エントリー価格
# =========================
g = df.groupby("Ticker")

df["entry_price"] = g["Open"].shift(-1)  # 翌日始値
df["exit_price"] = g["Close"].shift(-HOLD_DAYS)

# =========================
# リターン
# =========================
df["ret"] = (
    df["exit_price"] / df["entry_price"] - 1
)

df["ret"] -= COST * 2  # 売買コスト

# =========================
# トレード抽出
# =========================
trades = df[df["signal"]].copy()

# =========================
# 日次制限
# =========================
trades["rank"] = (
    trades.groupby("Date")["signal_score"]
    .rank(ascending=False, method="first")
)

trades = trades[trades["rank"] <= MAX_POSITIONS]

# =========================
# ポートフォリオ
# =========================
trades["weight"] = 1 / MAX_POSITIONS

# =========================
# 日次リターン作成
# =========================
daily_returns = []

dates = sorted(df["Date"].unique())

for date in dates:

    day_trades = trades[trades["Date"] == date]

    if len(day_trades) == 0:
        daily_returns.append(0)
        continue

    ret = (day_trades["ret"] * day_trades["weight"]).sum()

    daily_returns.append(ret)

# =========================
# 累積
# =========================
daily_returns = pd.Series(daily_returns, index=dates)

equity = (1 + daily_returns).cumprod()

# =========================
# 指標
# =========================
cagr = equity.iloc[-1] ** (252 / len(equity)) - 1

sharpe = (
    daily_returns.mean() /
    daily_returns.std()
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
# 保存（任意）
# =========================
equity.to_csv("equity_curve.csv")
trades.to_csv("trades.csv", index=False)
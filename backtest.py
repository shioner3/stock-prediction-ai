import pandas as pd
import numpy as np

from signal_engine import generate_signals

# =========================
# 設定
# =========================
FEATURE_PATH = "stock_data/features.parquet"

HOLD_DAYS = 5
MAX_POSITIONS = 3
INITIAL_CAPITAL = 1.0
COST = 0.001

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
# 価格
# =========================
g = df.groupby("Ticker")

df["entry_price"] = g["Open"].shift(-1)
df["exit_price"] = g["Close"].shift(-HOLD_DAYS)

# 有効トレードのみ
df = df.dropna(subset=["entry_price", "exit_price"])

# リターン
df["ret"] = df["exit_price"] / df["entry_price"] - 1 - COST * 2

# =========================
# トレード抽出
# =========================
trades = df[df["signal"]].copy()

if len(trades) == 0:
    print("❌ No trades generated")
    exit()

# ランキング制限
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
# ポジション展開
# =========================
positions = []

for _, row in trades.iterrows():

    entry_date = row["Date"]
    exit_date = entry_date + pd.Timedelta(days=HOLD_DAYS)

    # ★ 修正ポイント
    daily_ret = (1 + row["ret"]) ** (1 / HOLD_DAYS) - 1

    positions.append({
        "entry_date": entry_date,
        "exit_date": exit_date,
        "daily_ret": daily_ret
    })

pos_df = pd.DataFrame(positions)

# =========================
# 日次シミュレーション
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

    # 均等配分
    n = min(len(active), MAX_POSITIONS)

    daily_ret = active["daily_ret"].mean()

    capital *= (1 + daily_ret)

    equity_curve.append(capital)
    
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
trades.to_csv("trades.csv", index=False)
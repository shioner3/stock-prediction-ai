import pandas as pd
import numpy as np

from signal_engine import generate_signals

# =========================
# 設定
# =========================
FEATURE_PATH = "stock_data/features.parquet"

MAX_POSITIONS = 1
MAX_HOLD_DAYS = 15
INITIAL_CAPITAL = 1.0
COST = 0.001

# =========================
# データ
# =========================
df = pd.read_parquet(FEATURE_PATH)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

df = generate_signals(df)

print("Rows:", len(df))
print("Signals:", df["signal"].sum())

# =========================
# 便利辞書
# =========================
grouped = df.groupby("Ticker")
dates = sorted(df["Date"].unique())

# =========================
# ポジション管理
# =========================
positions = []   # active positions
closed_trades = []

capital = INITIAL_CAPITAL

# =========================
# イベントループ
# =========================
for date in dates:

    day_data = df[df["Date"] == date]

    # =========================
    # ① エグジット判定
    # =========================
    new_positions = []

    for pos in positions:

        df_t = grouped.get_group(pos["ticker"]).reset_index(drop=True)

        idx = df_t.index[df_t["Date"] == date]

        if len(idx) == 0:
            new_positions.append(pos)
            continue

        i = idx[0]

        entry_price = pos["entry_price"]

        current_price = df_t.iloc[i]["Close"]
        ret = current_price / entry_price - 1

        hold_days = (date - pos["entry_date"]).days

        exit_flag = False

        if ret > 0.15:
            exit_flag = True

        if (
            (df_t.iloc[i]["return_3d"] < -0.02) or
            (df_t.iloc[i]["ma5_diff"] < -0.03) or
            (df_t.iloc[i]["market_trend_5"] < -0.002)
        ):
            exit_flag = True

        if hold_days >= MAX_HOLD_DAYS:
            exit_flag = True

        if exit_flag:

            exit_price = df_t.iloc[i]["Close"]

            trade_ret = exit_price / entry_price - 1 - COST * 2

            capital *= (1 + trade_ret)

            closed_trades.append(trade_ret)

        else:
            new_positions.append(pos)

    positions = new_positions

    # =========================
    # ② エントリー判定
    # =========================
    candidates = day_data[day_data["signal"]].copy()

    if len(candidates) > 0 and len(positions) < MAX_POSITIONS:

        candidates["rank"] = candidates["signal_score"].rank(ascending=False)

        candidates = candidates.sort_values("rank")

        for _, row in candidates.iterrows():

            if len(positions) >= MAX_POSITIONS:
                break

            ticker = row["Ticker"]

            df_t = grouped.get_group(ticker).reset_index(drop=True)

            idx = df_t.index[df_t["Date"] == date]

            if len(idx) == 0:
                continue

            i = idx[0]

            if i + 1 >= len(df_t):
                continue

            entry_price = df_t.iloc[i + 1]["Open"]

            positions.append({
                "ticker": ticker,
                "entry_date": date,
                "entry_price": entry_price
            })

    # =========================
    # ③ equity更新（リアル）
    # =========================
    equity = capital

# =========================
# 最終評価
# =========================
closed_trades = np.array(closed_trades)

if len(closed_trades) == 0:
    print("❌ No trades executed")
    exit()

equity_curve = (1 + closed_trades).cumprod()

rets = pd.Series(equity_curve).pct_change().dropna()

cagr = equity_curve[-1] ** (252 / len(equity_curve)) - 1

sharpe = rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)

max_dd = (equity_curve / np.maximum.accumulate(equity_curve) - 1).min()

# =========================
# 出力
# =========================
print("\n=== EVENT DRIVEN RESULT ===")
print(f"CAGR  : {cagr:.4f}")
print(f"Sharpe: {sharpe:.4f}")
print(f"MaxDD : {max_dd:.4f}")
print(f"Trades: {len(closed_trades)}")
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

FEATURES = [
    "Return_1","Return_3",
    "Rank_Return_1",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio",
    "HL_range",
    "RSI",
    "Rel_Return_1",
    "Trend_5_z",
    "Trend_10_z",
    "Volatility_z",
    "Volume_ratio_z",
    "Market_Return_z"
]

TARGET = "Target"

INITIAL_CAPITAL = 1.0

THRESHOLD = 0.53

HOLD_DAYS = 7
STOP_LOSS = -0.03
TAKE_PROFIT = 0.10
MARKET_FILTER = -0.005

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

years = sorted(df["Date"].dt.year.unique())

# =========================
# バックテスト
# =========================
def run_backtest(train_df, test_df):

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        random_state=42
    )

    model.fit(train_df[FEATURES], train_df[TARGET])
    test_df = test_df.copy()
    test_df["pred"] = model.predict_proba(test_df[FEATURES])[:, 1]

    dates = sorted(test_df["Date"].unique())
    date_index = {d: i for i, d in enumerate(dates)}

    equity = INITIAL_CAPITAL
    equity_curve = []
    positions = []

    trade_count = 0

    for d in dates:

        today = test_df[test_df["Date"] == d]
        daily_pnl = 0

        # =========================
        # 決済
        # =========================
        new_positions = []
        for pos in positions:

            cur = today[today["Ticker"] == pos["ticker"]]

            if cur.empty:
                new_positions.append(pos)
                continue

            price = cur["Close"].iloc[0]
            ret = (price - pos["entry_price"]) / pos["entry_price"]

            if ret < STOP_LOSS or ret > TAKE_PROFIT or d >= pos["exit_date"]:
                daily_pnl += pos["capital"] * ret
            else:
                new_positions.append(pos)

        positions = new_positions

        # =========================
        # 相場フィルタ
        # =========================
        market = today["Return_1"].mean()
        if market < MARKET_FILTER:
            equity += daily_pnl
            equity_curve.append(equity)
            continue

        # =========================
        # エントリー候補
        # =========================
        today_f = today[today["pred"] > THRESHOLD]

        if not today_f.empty:

            # =========================
            # 🔥 ノートレ条件（追加）
            # =========================


            # =========================
            # 🔥 動的TOP_N
            # =========================


            # =========================
            # 🔥 weight = pred^2
            # =========================
            weights = picks["pred"] ** 2.2
            total_weight = weights.sum()

            invested = sum([p["capital"] for p in positions])
            free_cash = equity - invested

            if d not in date_index or date_index[d] + 1 >= len(dates):
                equity += daily_pnl
                equity_curve.append(equity)
                continue

            next_day = dates[date_index[d] + 1]
            next_data = test_df[test_df["Date"] == next_day]

            for i, (_, row) in enumerate(picks.iterrows()):

                if any(p["ticker"] == row["Ticker"] for p in positions):
                    continue

                next_row = next_data[next_data["Ticker"] == row["Ticker"]]
                if next_row.empty:
                    continue

                entry_price = next_row["Open"].iloc[0]

                weight = weights.iloc[i] / total_weight
                capital = free_cash * weight

                if capital <= 0:
                    continue

                positions.append({
                    "ticker": row["Ticker"],
                    "entry_price": entry_price,
                    "entry_date": next_day,
                    "exit_date": next_day + pd.Timedelta(days=HOLD_DAYS),
                    "capital": capital
                })

                trade_count += 1

        equity += daily_pnl
        equity_curve.append(equity)

    equity_curve = pd.Series(equity_curve)
    returns = equity_curve.pct_change().dropna()

    CAGR = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
    Sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
    MaxDD = (equity_curve / equity_curve.cummax() - 1).min()

    return CAGR, Sharpe, MaxDD, trade_count


# =========================
# 実行
# =========================
results = []

for test_year in years:

    if test_year < 2022:
        continue

    print(f"\n=== YEAR {test_year} ===")

    train_df = df[df["Date"].dt.year < test_year]
    test_df = df[df["Date"].dt.year == test_year]

    CAGR, Sharpe, MaxDD, trades = run_backtest(train_df, test_df)

    print(f"CAGR: {CAGR:.3f}")
    print(f"Sharpe: {Sharpe:.3f}")
    print(f"MaxDD: {MaxDD:.3f}")
    print(f"Trades: {trades}")

    results.append({
        "year": test_year,
        "CAGR": CAGR,
        "Sharpe": Sharpe,
        "MaxDD": MaxDD,
        "Trades": trades
    })

# =========================
# 集計
# =========================
df_res = pd.DataFrame(results)

print("\n=== SUMMARY ===")
print(df_res)

print("\n平均")
print(df_res.mean(numeric_only=True))
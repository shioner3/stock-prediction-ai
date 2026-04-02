import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

DATA_PATH = "ml_dataset.parquet"

FEATURES = [
    "Return_1","Return_3","Return_5",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio","Volume_accel",
    "HL_range",
    "EMA_gap",
    "Momentum_5","Momentum_10",
    "ATR_ratio"
]

TARGET = "Target"

INITIAL_CAPITAL = 1.0

THRESHOLD = 0.28
HOLD_DAYS = 7
STOP_LOSS = -0.02
TAKE_PROFIT = 0.08

df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

years = sorted(df["Date"].dt.year.unique())
results = []

for test_year in years:

    if test_year < 2022:
        continue

    print(f"\n=== WALK FORWARD: {test_year} ===")

    train_df = df[df["Date"].dt.year < test_year]
    test_df = df[df["Date"].dt.year == test_year].copy()

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(train_df[FEATURES], train_df[TARGET])

    test_df["pred"] = model.predict_proba(test_df[FEATURES])[:, 1]

    def make_hybrid_score(df):
        df = df.copy()
        df["pred_rank"] = df["pred"].rank(pct=True)
        df["mom_rank"] = df["Return_5"].rank(pct=True)
        df["trend_rank"] = df["EMA_gap"].rank(pct=True)

        df["hybrid_score"] = (
            0.5 * df["pred_rank"] +
            0.3 * df["mom_rank"] +
            0.2 * df["trend_rank"]
        )
        return df

    dates = sorted(test_df["Date"].unique())
    date_index = {d: i for i, d in enumerate(dates)}

    equity = 1.0
    positions = []
    equity_curve = []
    trade_count = 0

    for d in dates:

        today = test_df[test_df["Date"] == d]
        daily_pnl = 0

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

        today_f = today[today["pred"] > THRESHOLD]
        today_f = today_f[today_f["EMA_gap"] > 0]

        if not today_f.empty:

            today_f = make_hybrid_score(today_f)
            picks = today_f.sort_values("hybrid_score", ascending=False).head(2)

            total_pred = picks["pred"].sum()

            if total_pred > 0:

                free_cash = equity - sum([p["capital"] for p in positions])

                if d not in date_index or date_index[d] + 1 >= len(dates):
                    equity += daily_pnl
                    equity_curve.append(equity)
                    continue

                next_day = dates[date_index[d] + 1]
                next_data = test_df[test_df["Date"] == next_day]

                for _, row in picks.iterrows():

                    next_row = next_data[next_data["Ticker"] == row["Ticker"]]
                    if next_row.empty:
                        continue

                    entry_price = next_row["Open"].iloc[0]

                    weight = row["pred"] / total_pred
                    capital = free_cash * weight

                    if capital <= 0:
                        continue

                    positions.append({
                        "ticker": row["Ticker"],
                        "entry_price": entry_price,
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

    print(f"CAGR: {CAGR:.3f}")
    print(f"Sharpe: {Sharpe:.3f}")
    print(f"MaxDD: {MaxDD:.3f}")
    print(f"Trades: {trade_count}")

    results.append({
        "year": test_year,
        "CAGR": CAGR,
        "Sharpe": Sharpe,
        "MaxDD": MaxDD,
        "Trades": trade_count
    })

df_res = pd.DataFrame(results)

print("\n=== SUMMARY ===")
print(df_res)

print("\n平均")
print(df_res.mean(numeric_only=True))
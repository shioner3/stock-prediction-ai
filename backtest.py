import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

FEATURES = [
    "Return_1","Return_3",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio",
    "HL_range","RSI"
]

TARGET = "Target"

INITIAL_CAPITAL = 1.0

THRESHOLD = 0.55   # ←重要（分類に変えたので上げる）
HOLD_DAYS = 7
STOP_LOSS = -0.03
TAKE_PROFIT = 0.10

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

years = sorted(df["Date"].dt.year.unique())
results = []

# =========================
# ウォークフォワード
# =========================
for test_year in years:

    if test_year < 2022:
        continue

    print(f"\n=== WALK FORWARD: {test_year} ===")

    train_df = df[df["Date"].dt.year < test_year]
    test_df = df[df["Date"].dt.year == test_year].copy()

    if len(test_df) == 0:
        continue

    # =========================
    # 学習
    # =========================
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42
    )

    model.fit(train_df[FEATURES], train_df[TARGET])

    test_df["pred"] = model.predict_proba(test_df[FEATURES])[:, 1]

    # =========================
    # 日付
    # =========================
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
                pnl = pos["capital"] * ret
                daily_pnl += pnl
            else:
                new_positions.append(pos)

        positions = new_positions

        # =========================
        # エントリー
        # =========================
        today_f = today[today["pred"] > THRESHOLD]

        if not today_f.empty:

            # 🔥 市場フィルタ（シンプル版）
            market = today_f["Return_1"].mean()

            if market < -0.01:
                equity += daily_pnl
                equity_curve.append(equity)
                continue

            # 🔥 上位2銘柄のみ
            picks = today_f.sort_values("pred", ascending=False).head(2)

            total_pred = picks["pred"].sum()

            invested = sum([p["capital"] for p in positions])
            free_cash = equity - invested

            if d not in date_index or date_index[d] + 1 >= len(dates):
                equity += daily_pnl
                equity_curve.append(equity)
                continue

            next_day = dates[date_index[d] + 1]
            next_data = test_df[test_df["Date"] == next_day]

            for _, row in picks.iterrows():

                if any(p["ticker"] == row["Ticker"] for p in positions):
                    continue

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
                    "entry_date": next_day,
                    "exit_date": next_day + pd.Timedelta(days=HOLD_DAYS),
                    "capital": capital
                })

                trade_count += 1

        equity += daily_pnl
        equity_curve.append(equity)

    # =========================
    # 評価
    # =========================
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

# =========================
# 集計
# =========================
df_res = pd.DataFrame(results)

print("\n=== SUMMARY ===")
print(df_res)

print("\n平均")
print(df_res.mean(numeric_only=True))
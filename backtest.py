import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import itertools

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

FEATURES = [
    "Return_1","Return_3",
    "Rank_Return_1","Rank_Volume",
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
HOLD_DAYS = 7

STOP_LOSS = -0.03
TAKE_PROFIT = 0.10

# =========================
# 探索パラメータ
# =========================
THRESHOLDS = [0.50, 0.52, 0.54, 0.56, 0.58]
TOP_N_LIST = [1, 2, 3]
MARKET_FILTERS = [-0.01, -0.005, -0.003, 0]

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

years = sorted(df["Date"].dt.year.unique())

# =========================
# 🔥 年1回だけ学習（ここが重要）
# =========================
train_cutoff_year = 2021
train_df = df[df["Date"].dt.year <= train_cutoff_year]

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    random_state=42
)

print("Training model once...")
model.fit(train_df[FEATURES], train_df[TARGET])
print("Done.")

# =========================
# バックテスト関数（学習なし）
# =========================
def run_backtest(test_df, THRESHOLD, TOP_N, MARKET_FILTER):

    test_df = test_df.copy()
    test_df["pred"] = model.predict_proba(test_df[FEATURES])[:, 1]

    dates = sorted(test_df["Date"].unique())
    date_index = {d: i for i, d in enumerate(dates)}

    equity = INITIAL_CAPITAL
    equity_curve = []
    positions = []

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
        # Market Filter
        # =========================
        market = today["Return_1"].mean()
        if market < MARKET_FILTER:
            equity += daily_pnl
            equity_curve.append(equity)
            continue

        # =========================
        # エントリー
        # =========================
        today_f = today[today["pred"] > THRESHOLD]

        if today_f.empty:
            equity += daily_pnl
            equity_curve.append(equity)
            continue

        picks = today_f.sort_values("pred", ascending=False).head(TOP_N)

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

        equity += daily_pnl
        equity_curve.append(equity)

    equity_curve = pd.Series(equity_curve)

    if len(equity_curve) < 10:
        return None

    returns = equity_curve.pct_change().dropna()

    CAGR = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
    Sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
    MaxDD = (equity_curve / equity_curve.cummax() - 1).min()

    return CAGR, Sharpe, MaxDD

# =========================
# 🔥 最適化
# =========================
results = []

for TH, TN, MF in itertools.product(THRESHOLDS, TOP_N_LIST, MARKET_FILTERS):

    print(f"\n=== TH={TH}, TOP_N={TN}, MF={MF} ===")

    yearly = []

    for test_year in years:
        if test_year < 2022:
            continue

        test_df = df[df["Date"].dt.year == test_year]

        res = run_backtest(test_df, TH, TN, MF)

        if res is None:
            continue

        yearly.append(res)

    if len(yearly) == 0:
        continue

    CAGR = np.mean([x[0] for x in yearly])
    Sharpe = np.mean([x[1] for x in yearly])
    MaxDD = np.mean([x[2] for x in yearly])

    results.append({
        "THRESHOLD": TH,
        "TOP_N": TN,
        "MARKET_FILTER": MF,
        "CAGR": CAGR,
        "Sharpe": Sharpe,
        "MaxDD": MaxDD
    })

# =========================
# 結果
# =========================
df_res = pd.DataFrame(results)

df_res = df_res.sort_values("Sharpe", ascending=False)

print("\n=== BEST PARAMS ===")
print(df_res.head(10))
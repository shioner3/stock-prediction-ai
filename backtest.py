import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

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
# 固定パラメータ
# =========================
THRESHOLD = 0.52
TOP_N_NORMAL = 1
TOP_N_RANGE = 1

# レンジ時の調整
RANGE_THRESHOLD_MULTIPLIER = 1.05   # 閾値を少し上げる
RANGE_CAPITAL_RATIO = 0.5          # 資金半分

# =========================
# レジーム設定
# =========================
MARKET_FILTER = -0.003
MARKET_MA_WINDOW = 20

# =========================
# レンジ検出
# =========================
RANGE_WINDOW = 20
RANGE_STD_THRESHOLD = 0.006

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

years = sorted(df["Date"].dt.year.unique())

# =========================
# 学習
# =========================
train_cutoff_year = 2021
train_df = df[df["Date"].dt.year <= train_cutoff_year]

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    random_state=42
)

print("Training model...")
model.fit(train_df[FEATURES], train_df[TARGET])
print("Done.")

# =========================
# バックテスト
# =========================
def run_backtest(test_df):

    test_df = test_df.copy()
    test_df["pred"] = model.predict_proba(test_df[FEATURES])[:, 1]

    dates = sorted(test_df["Date"].unique())
    date_index = {d: i for i, d in enumerate(dates)}

    equity = INITIAL_CAPITAL
    equity_curve = []
    positions = []

    market_daily = df.groupby("Date")["Return_1"].mean()

    for i, d in enumerate(dates):

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
        # レジーム判定
        # =========================
        market_ret = today["Return_1"].mean()

        if i >= MARKET_MA_WINDOW:
            past_dates = dates[i-MARKET_MA_WINDOW:i]
            past_market = df[df["Date"].isin(past_dates)]
            market_ma = past_market["Return_1"].mean()
        else:
            market_ma = 0

        # =========================
        # レンジ検出
        # =========================
        if d in market_daily.index:
            idx = market_daily.index.get_loc(d)

            if idx >= RANGE_WINDOW:
                past_vals = market_daily.iloc[idx-RANGE_WINDOW:idx]
                market_std = past_vals.std()
            else:
                market_std = 1.0
        else:
            market_std = 1.0

        is_range = market_std < RANGE_STD_THRESHOLD

        # =========================
        # モード決定
        # =========================
        if is_range:
            top_n = TOP_N_RANGE
            threshold = THRESHOLD * RANGE_THRESHOLD_MULTIPLIER
            capital_ratio = RANGE_CAPITAL_RATIO
        else:
            top_n = TOP_N_NORMAL
            threshold = THRESHOLD
            capital_ratio = 1.0

        # =========================
        # レジームフィルター（弱）
        # =========================
        if market_ret < MARKET_FILTER or market_ma < 0:
            equity += daily_pnl
            equity_curve.append(equity)
            continue

        # =========================
        # エントリー
        # =========================
        today_f = today[today["pred"] > threshold]

        if today_f.empty:
            equity += daily_pnl
            equity_curve.append(equity)
            continue

        picks = today_f.sort_values("pred", ascending=False).head(top_n)

        invested = sum([p["capital"] for p in positions])
        free_cash = (equity - invested) * capital_ratio

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

            positions.append({
                "ticker": row["Ticker"],
                "entry_price": entry_price,
                "entry_date": next_day,
                "exit_date": next_day + pd.Timedelta(days=HOLD_DAYS),
                "capital": free_cash
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
# 年別テスト
# =========================
print("\n=== REPRODUCIBILITY TEST (REGIME + RANGE SCALED) ===")

results = []

for year in years:
    if year < 2022:
        continue

    print(f"\n--- Year {year} ---")

    test_df = df[df["Date"].dt.year == year]

    res = run_backtest(test_df)

    if res is None:
        print("Skipped")
        continue

    CAGR, Sharpe, MaxDD = res

    results.append({
        "Year": year,
        "CAGR": CAGR,
        "Sharpe": Sharpe,
        "MaxDD": MaxDD
    })

    print(f"CAGR={CAGR:.3f}, Sharpe={Sharpe:.3f}, MaxDD={MaxDD:.3f}")

# =========================
# 集計
# =========================
df_res = pd.DataFrame(results)

print("\n=== SUMMARY ===")
print(df_res)

print("\n=== AVERAGE ===")
print(df_res.mean(numeric_only=True))
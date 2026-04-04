import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from hmmlearn.hmm import GaussianHMM

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
HOLD_DAYS = 10

STOP_LOSS = -0.03
TAKE_PROFIT = 0.12

CRASH_FILTER = -0.02

HIGH_VOL = 2.0
MID_VOL = 1.0

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])
df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf]).fillna(0)

# =========================
# HMM（TREND抽出だけ使う）
# =========================
HMM_FEATURES = ["Return_1", "Volatility", "Volume_ratio"]

market = df.groupby("Date")[HMM_FEATURES].mean().reset_index()

hmm = GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=200,
    random_state=42
)

hmm.fit(market[HMM_FEATURES])
proba = hmm.predict_proba(market[HMM_FEATURES])

state = np.argmax(proba, axis=1)
market["state"] = state

state_ret = {}
for s in range(3):
    state_ret[s] = market.loc[market["state"] == s, "Return_1"].mean()

TREND_STATE = max(state_ret.items(), key=lambda x: x[1])[0]

market_map = market.set_index("Date")[["state"]]

# =========================
# モデル（全データでOK）
# =========================
train_df = df[df["Date"].dt.year <= 2021]

model = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=6,
    random_state=42
)

print("Training TREND model...")
model.fit(train_df[FEATURES], train_df[TARGET])
print("Done.")

# =========================
# バックテスト（TRENDのみ）
# =========================
def run_backtest(test_df):

    test_df = test_df.copy()
    test_df["pred"] = model.predict_proba(test_df[FEATURES])[:, 1]

    dates = sorted(test_df["Date"].unique())
    date_index = {d: i for i, d in enumerate(dates)}

    test_df = test_df.merge(market_map, left_on="Date", right_index=True, how="left")

    equity = INITIAL_CAPITAL
    equity_curve = []
    positions = []

    for i, d in enumerate(dates):

        today = test_df[test_df["Date"] == d]

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
                equity += pos["capital"] * ret
            else:
                new_positions.append(pos)

        positions = new_positions

        # =========================
        # CRASHフィルタ
        # =========================
        market_ret = today["Return_1"].mean()

        if market_ret < CRASH_FILTER:
            equity_curve.append(equity)
            continue

        # =========================
        # TREND以外は取引しない
        # =========================
        s = today["state"].iloc[0]

        if s != TREND_STATE:
            equity_curve.append(equity)
            continue

        # =========================
        # ボラ調整
        # =========================
        vol = today["Volatility"].mean()

        if vol > HIGH_VOL:
            cap_ratio = 0.4
        elif vol > MID_VOL:
            cap_ratio = 0.7
        else:
            cap_ratio = 1.0

        # =========================
        # エントリー
        # =========================
        today_f = today[today["pred"] > 0.52]

        if today_f.empty:
            equity_curve.append(equity)
            continue

        picks = today_f.sort_values("pred", ascending=False).head(5)

        invested = sum([p["capital"] for p in positions])
        free_cash = (equity - invested) * cap_ratio

        if i + 1 >= len(dates):
            equity_curve.append(equity)
            continue

        next_day = dates[i + 1]
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
                "capital": free_cash / 5
            })

        equity_curve.append(equity)

    equity_curve = pd.Series(equity_curve)

    returns = equity_curve.pct_change().dropna()

    CAGR = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
    Sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
    MaxDD = (equity_curve / equity_curve.cummax() - 1).min()

    return CAGR, Sharpe, MaxDD


# =========================
# テスト
# =========================
print("\n=== TREND ONLY MODEL ===")

results = []

for year in sorted(df["Date"].dt.year.unique()):
    if year < 2022:
        continue

    print(f"\n--- Year {year} ---")

    test_df = df[df["Date"].dt.year == year]

    CAGR, Sharpe, MaxDD = run_backtest(test_df)

    results.append({
        "Year": year,
        "CAGR": CAGR,
        "Sharpe": Sharpe,
        "MaxDD": MaxDD
    })

df_res = pd.DataFrame(results)

print("\n=== SUMMARY ===")
print(df_res)

print("\n=== AVERAGE ===")
print(df_res.mean(numeric_only=True))
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
HOLD_DAYS = 7

STOP_LOSS = -0.03
TAKE_PROFIT = 0.10

THRESHOLD = 0.52

# =========================
# 🆕 追加（クラッシュ＋TOP）
# =========================
CRASH_FILTER = -0.02
TOP_N = 2

# =========================
# 🆕 ボラティリティ閾値
# =========================
HIGH_VOL = 2.0
MID_VOL = 1.0

# =========================
# 手作りレジーム
# =========================
MARKET_FILTER = -0.003
MARKET_MA_WINDOW = 20

# =========================
# HMM設定
# =========================
HMM_FEATURES = ["Return_1", "Volatility", "Volume_ratio"]
HMM_SKIP = 0.75
HMM_WEAK = 0.55

# =========================
# DD STOP
# =========================
MAX_DRAWDOWN = -0.20

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf]).fillna(0)

# =========================
# 市場データ（HMM）
# =========================
market = df.groupby("Date")[HMM_FEATURES].mean().reset_index()

hmm = GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=200,
    random_state=42
)

hmm.fit(market[HMM_FEATURES])
proba = hmm.predict_proba(market[HMM_FEATURES])

state_ret = {}
for i in range(3):
    mask = proba[:, i] > 0.5
    state_ret[i] = market["Return_1"][mask].mean() if mask.sum() > 0 else 0

sorted_states = sorted(state_ret.items(), key=lambda x: x[1])
DOWN_STATE = sorted_states[0][0]

market["down_p"] = proba[:, DOWN_STATE]

# =========================
# 学習
# =========================
train_df = df[df["Date"].dt.year <= 2021]

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

    market_map = market.set_index("Date")[["down_p"]]
    test_df = test_df.merge(market_map, left_on="Date", right_index=True, how="left")

    equity = INITIAL_CAPITAL
    equity_curve = []
    positions = []

    peak_equity = INITIAL_CAPITAL
    trading_stopped = False

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
        equity += daily_pnl

        # =========================
        # DD管理
        # =========================
        peak_equity = max(peak_equity, equity)
        dd = (equity / peak_equity) - 1

        if dd < MAX_DRAWDOWN:
            trading_stopped = True

        if trading_stopped and dd > -0.05:
            trading_stopped = False

        if trading_stopped:
            equity_curve.append(equity)
            continue

        # =========================
        # クラッシュフィルタ
        # =========================
        market_ret = today["Return_1"].mean()

        if market_ret < CRASH_FILTER:
            equity_curve.append(equity)
            continue

        # =========================
        # 手作りレジーム
        # =========================
        if i >= MARKET_MA_WINDOW:
            past_dates = dates[i-MARKET_MA_WINDOW:i]
            past_market = df[df["Date"].isin(past_dates)]
            market_ma = past_market["Return_1"].mean()
        else:
            market_ma = 0

        manual_down = (market_ret < MARKET_FILTER) or (market_ma < 0)

        # =========================
        # HMM
        # =========================
        down_p = today["down_p"].iloc[0] if not today.empty else 0

        if manual_down or down_p > HMM_SKIP:
            equity_curve.append(equity)
            continue

        # =========================
        # 🆕 ボラティリティベース資金管理
        # =========================
        volatility = today["Volatility"].mean()

        if volatility > HIGH_VOL:
            capital_ratio = 0.3
        elif volatility > MID_VOL:
            capital_ratio = 0.6
        else:
            capital_ratio = 1.0

        # =========================
        # エントリー
        # =========================
        today_f = today[today["pred"] > THRESHOLD]

        if today_f.empty:
            equity_curve.append(equity)
            continue

        picks = today_f.sort_values("pred", ascending=False).head(TOP_N)

        invested = sum([p["capital"] for p in positions])
        free_cash = (equity - invested) * capital_ratio

        if d not in date_index or date_index[d] + 1 >= len(dates):
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
                "capital": free_cash / TOP_N
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
print("\n=== FINAL HYBRID + VOL CONTROL ===")

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

    print(f"CAGR={CAGR:.3f}, Sharpe={Sharpe:.3f}, MaxDD={MaxDD:.3f}")

df_res = pd.DataFrame(results)

print("\n=== SUMMARY ===")
print(df_res)

print("\n=== AVERAGE ===")
print(df_res.mean(numeric_only=True))
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
# HMM特徴
# =========================
HMM_FEATURES = ["Return_1", "Volatility", "Volume_ratio"]

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

# =========================
# 市場データ（HMM用）
# =========================
market = df.groupby("Date")[HMM_FEATURES].mean().reset_index()

# =========================
# HMM学習
# =========================
hmm = GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=200,
    random_state=42
)

hmm.fit(market[HMM_FEATURES])

# ★ ここが重要（確率を使う）
proba = hmm.predict_proba(market[HMM_FEATURES])

market["down_p"] = proba[:, 0]
market["mid_p"] = proba[:, 1]
market["trend_p"] = proba[:, 2]

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

    # =========================
    # HMMマージ
    # =========================
    market_map = market.set_index("Date")[["down_p","mid_p","trend_p"]]

    test_df = test_df.merge(
        market_map,
        left_on="Date",
        right_index=True,
        how="left"
    )

    # =========================
    # スコア調整（ここが本体）
    # =========================
    test_df["regime_score"] = (
        test_df["trend_p"] * 1.0 +
        test_df["mid_p"] * 0.0 +
        test_df["down_p"] * (-1.0)
    )

    test_df["adj_pred"] = test_df["pred"] * (1 + 0.5 * test_df["regime_score"])

    equity = INITIAL_CAPITAL
    equity_curve = []
    positions = []

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
        # エントリー
        # =========================
        today_f = today[today["adj_pred"] > THRESHOLD]

        if today_f.empty:
            equity += daily_pnl
            equity_curve.append(equity)
            continue

        picks = today_f.sort_values("adj_pred", ascending=False).head(1)

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

    returns = equity_curve.pct_change().dropna()

    CAGR = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
    Sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
    MaxDD = (equity_curve / equity_curve.cummax() - 1).min()

    return CAGR, Sharpe, MaxDD


# =========================
# テスト
# =========================
print("\n=== HMM SCORE-ADJUSTED BACKTEST ===")

results = []

for year in sorted(df["Date"].dt.year.unique()):
    if year < 2022:
        continue

    print(f"\n--- Year {year} ---")

    test_df = df[df["Date"].dt.year == year]

    res = run_backtest(test_df)

    CAGR, Sharpe, MaxDD = res

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
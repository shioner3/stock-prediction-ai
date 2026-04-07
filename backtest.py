import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

INITIAL_CAPITAL = 1.0
MAX_POSITIONS = 10
HOLD_DAYS = 5

N_BINS = 10
THRESHOLD = 0.002  # 🔥 重要（調整ポイント）

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

FEATURES = [
    "Return_1","Return_3",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio",
    "HL_range",
    "Rel_Return_1",
    "Trend_5_z","Trend_10_z",
    "Gap","Volatility_change","Volume_spike","Momentum_acc"
]

# =========================
# レジーム
# =========================
market = df.groupby("Date")["Return_1"].mean()
market_smooth = market.rolling(20).mean()

df["Market_Smooth"] = df["Date"].map(market_smooth)

df["Regime"] = np.where(
    df["Market_Smooth"] > 0.001, "up",
    np.where(df["Market_Smooth"] < -0.001, "down", "range")
)

# =========================
# モデル
# =========================
def train_model(train_df):
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(train_df[FEATURES], train_df["Target"])
    return model

# =========================
# バックテスト
# =========================
def run_backtest(train_df, test_df):

    model = train_model(train_df)

    test_df = test_df.copy()
    test_df["raw_score"] = model.predict(test_df[FEATURES])

    dates = sorted(test_df["Date"].unique())
    grouped = {d: g for d, g in test_df.groupby("Date")}

    equity = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    equity_curve = []

    positions = []

    for i, d in enumerate(dates):

        today = grouped[d]

        # =========================
        # 決済
        # =========================
        new_positions = []

        for pos in positions:
            if i == pos["exit_idx"]:

                cur = today[today["Ticker"] == pos["ticker"]]
                if cur.empty:
                    continue

                exit_price = cur["Open"].iloc[0]
                ret = (exit_price - pos["entry_price"]) / pos["entry_price"]

                cash += pos["capital"] * (1 + ret)

            else:
                new_positions.append(pos)

        positions = new_positions

        # =========================
        # エントリー（🔥ハイブリッド）
        # =========================
        if i + 1 < len(dates):

            available = MAX_POSITIONS - len(positions)

            if available > 0 and cash > 0:

                next_day = dates[i + 1]
                next_data = grouped[next_day]

                # =========================
                # ① 逆張り
                # =========================
                mr = today[
                    (today["Return_1"] < -0.02) &
                    (today["raw_score"] > THRESHOLD)
                ]

                # =========================
                # ② 順張り
                # =========================
                mom = today[
                    (today["Return_1"] > 0.01) &
                    (today["raw_score"] > THRESHOLD)
                ]

                # =========================
                # 上位抽出
                # =========================
                mr_picks = mr.sort_values("raw_score", ascending=False).head(available // 2)
                mom_picks = mom.sort_values("raw_score", ascending=False).head(available // 2)

                picks = pd.concat([mr_picks, mom_picks])

                if len(picks) > 0:

                    capital_per_position = cash / len(picks)

                    for row in picks.itertuples():

                        ticker = row.Ticker

                        if any(p["ticker"] == ticker for p in positions):
                            continue

                        next_row = next_data[next_data["Ticker"] == ticker]
                        if next_row.empty:
                            continue

                        exit_idx = i + HOLD_DAYS
                        if exit_idx >= len(dates):
                            continue

                        positions.append({
                            "ticker": ticker,
                            "entry_price": next_row["Open"].iloc[0],
                            "exit_idx": exit_idx,
                            "capital": capital_per_position
                        })

                        cash -= capital_per_position

        # =========================
        # 評価
        # =========================
        pos_val = 0

        for pos in positions:
            cur = today[today["Ticker"] == pos["ticker"]]
            if not cur.empty:
                price = cur["Close"].iloc[0]
                ret = (price - pos["entry_price"]) / pos["entry_price"]
                pos_val += pos["capital"] * (1 + ret)

        equity = cash + pos_val
        equity_curve.append(equity)

    # =========================
    # 評価指標
    # =========================
    equity_curve = pd.Series(equity_curve)
    returns = equity_curve.pct_change().dropna()

    CAGR = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
    Sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
    MaxDD = (equity_curve / equity_curve.cummax() - 1).min()

    return (CAGR, Sharpe, MaxDD)

# =========================
# 実行
# =========================
all_metrics = []

for y in sorted(df["Date"].dt.year.unique()):

    if y < 2022:
        continue

    train_df = df[df["Date"].dt.year < y]
    test_df = df[df["Date"].dt.year == y]

    res = run_backtest(train_df, test_df)

    if res is not None:
        all_metrics.append(res)

# =========================
# 結果
# =========================
if len(all_metrics) > 0:

    cagr = np.mean([m[0] for m in all_metrics])
    sharpe = np.mean([m[1] for m in all_metrics])
    mdd = np.mean([m[2] for m in all_metrics])

    print("\n=== RESULT ===")
    print(f"CAGR  : {cagr:.4f}")
    print(f"Sharpe: {sharpe:.4f}")
    print(f"MaxDD : {mdd:.4f}")
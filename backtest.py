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
MIN_TICKERS = 3000

TOP_N = 5
MAX_POSITIONS = 5
HOLD_DAYS = 5  # 🔥 feature側と完全一致

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

print("\n=== TARGET CHECK ===")
print("Target mean:", df["Target"].mean())

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

# =========================
# 完全性チェック
# =========================
counts = df["Date"].value_counts()
valid_dates = counts[counts >= MIN_TICKERS].index
latest_valid_date = valid_dates.max()

print("\n=== DATA CHECK ===")
print("最新日:", df["Date"].max())
print("有効最新日:", latest_valid_date)

df = df[df["Date"] <= latest_valid_date]

# =========================
# OOS分割
# =========================
OOS_START = 2024

train_df_full = df[df["Date"].dt.year < OOS_START]
test_df_oos = df[df["Date"].dt.year >= OOS_START]

print("\n=== OOS TEST ===")
print("Train:", train_df_full["Date"].min(), "~", train_df_full["Date"].max())
print("Test :", test_df_oos["Date"].min(), "~", test_df_oos["Date"].max())

# =========================
# バックテスト
# =========================
def run_backtest(test_df, train_df, label="BASE"):

    equity = INITIAL_CAPITAL
    equity_curve = []

    model = None
    positions = []
    trade_count = 0

    dates = sorted(test_df["Date"].unique())
    prev_month = None

    for j in range(len(dates) - 1):

        d = dates[j]
        next_d = dates[j + 1]

        today = test_df[test_df["Date"] == d].copy()
        tomorrow = test_df[test_df["Date"] == next_d].copy()

        if today.empty:
            equity_curve.append(equity)
            continue

        # =========================
        # 月次学習
        # =========================
        current_month = d.month

        if model is None or current_month != prev_month:

            train_until = train_df[train_df["Date"] < d]

            if len(train_until) > 2000:
                model = LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    random_state=42
                )
                model.fit(train_until[FEATURES], train_until[TARGET])

            prev_month = current_month

        if model is None:
            equity_curve.append(equity)
            continue

        # =========================
        # 予測（確率）
        # =========================
        today_pred = today.copy()
        today_pred["pred"] = model.predict_proba(today_pred[FEATURES])[:, 1]

        # =========================
        # フィルター
        # =========================
        if "limit_up_flag" in today_pred.columns:
            today_pred = today_pred[today_pred["limit_up_flag"] == 0]

        if "Volume" in today_pred.columns:
            today_pred = today_pred[today_pred["Volume"].fillna(0) > 10000]

        if today_pred.empty:
            equity_curve.append(equity)
            continue

        # =========================
        # 🔥 Top N
        # =========================
        picks = today_pred.sort_values("pred", ascending=False).head(TOP_N)

        # =========================
        # エントリー
        # =========================
        if len(positions) < MAX_POSITIONS:

            for _, row in picks.iterrows():

                if len(positions) >= MAX_POSITIONS:
                    break

                if any(p["ticker"] == row["Ticker"] for p in positions):
                    continue

                tmr = tomorrow[tomorrow["Ticker"] == row["Ticker"]]
                if tmr.empty:
                    continue

                positions.append({
                    "ticker": row["Ticker"],
                    "entry_price": tmr["Open"].iloc[0],
                    "entry_day": j,
                    "weight": 1.0 / MAX_POSITIONS
                })

                trade_count += 1

        # =========================
        # ポジション管理
        # =========================
        new_positions = []

        for pos in positions:

            cur = today[today["Ticker"] == pos["ticker"]]

            if cur.empty:
                new_positions.append(pos)
                continue

            price = cur["Close"].iloc[0]
            ret = (price - pos["entry_price"]) / pos["entry_price"]

            hold_days_now = j - pos["entry_day"] + 1

            if hold_days_now >= HOLD_DAYS:
                equity *= (1 + ret * pos["weight"])
                continue

            new_positions.append(pos)

        positions = new_positions
        equity_curve.append(equity)

    equity_curve = pd.Series(equity_curve)

    if len(equity_curve) < 2:
        return None

    returns = equity_curve.pct_change().dropna()

    return {
        "label": label,
        "CAGR": equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1,
        "Sharpe": returns.mean() / (returns.std() + 1e-9) * np.sqrt(252),
        "MaxDD": (equity_curve / equity_curve.cummax() - 1).min(),
        "Trades": trade_count
    }

# =========================
# 実行
# =========================
base_result = run_backtest(test_df_oos, train_df_full, "BASE")

tickers = test_df_oos["Ticker"].unique()
reduced = np.random.choice(tickers, int(len(tickers)*0.7), replace=False)
test_reduced = test_df_oos[test_df_oos["Ticker"].isin(reduced)]

robust_1 = run_backtest(test_reduced, train_df_full, "Ticker70%")

dates = sorted(test_df_oos["Date"].unique())
cut = int(len(dates)*0.8)
test_short = test_df_oos[test_df_oos["Date"].isin(dates[:cut])]

robust_2 = run_backtest(test_short, train_df_full, "Time80%")

results = pd.DataFrame([base_result, robust_1, robust_2])

print("\n=== ROBUST TEST ===")
print(results)
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

FEATURES = [
    "Return_1","Return_3","Return_5",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio","Volume_accel",
    "HL_range",
    "EMA_gap",
    "Momentum_5","Momentum_10",
    "ATR_ratio",
    "RSI"
]

TARGET = "Target"

INITIAL_CAPITAL = 1.0

THRESHOLD = 0.26
HOLD_DAYS = 7
STOP_LOSS = -0.03

# 🔥 最適化対象
TAKE_PROFIT_LIST = [0.06, 0.08, 0.10, 0.12, 0.15]

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

# =========================
# 学習
# =========================
train_df = df[df["Date"].dt.year < 2024]
test_df = df[df["Date"].dt.year >= 2024].copy()

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

model.fit(train_df[FEATURES], train_df[TARGET])

# =========================
# 予測
# =========================
test_df["pred"] = model.predict_proba(test_df[FEATURES])[:, 1]

# =========================
# ハイブリッド
# =========================
def make_hybrid_score(df):
    df = df.copy()

    df["mom_rank"] = df["Return_5"].rank(pct=True)
    df["trend_rank"] = df["EMA_gap"].rank(pct=True)
    df["vol_rank"] = (-df["Volatility"]).rank(pct=True)

    df["hybrid_score"] = (
        0.5 * df["mom_rank"] +
        0.3 * df["trend_rank"] +
        0.2 * df["vol_rank"]
    )

    return df

# =========================
# バックテスト関数
# =========================
def run_backtest(take_profit):

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

            exit_flag = False

            if ret < STOP_LOSS:
                exit_flag = True
            elif ret > take_profit:
                exit_flag = True
            elif d >= pos["exit_date"]:
                exit_flag = True

            if exit_flag:
                pnl = pos["capital"] * ret
                daily_pnl += pnl
            else:
                new_positions.append(pos)

        positions = new_positions

        # =========================
        # エントリー
        # =========================
        today_f = today.copy()
        today_f = today_f[today_f["pred"] > THRESHOLD]
        today_f = today_f[today_f["EMA_gap"] > 0]

        if not today_f.empty:

            market = today_f["Return_1"].mean()
            market_pred_mean = today_f["pred"].mean()

            if market < -0.02:
                weight_cap = 0.2
                top_n = 1
            elif market < -0.01 or market_pred_mean < 0.30:
                weight_cap = 0.3
                top_n = 1
            else:
                weight_cap = 0.4
                top_n = 3

            today_f = make_hybrid_score(today_f)
            picks = today_f.sort_values("hybrid_score", ascending=False).head(top_n)

            total_pred = picks["pred"].sum()

            if total_pred > 0:

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

                    weight = min(row["pred"] / total_pred, weight_cap)
                    capital = free_cash * weight

                    if capital <= 0:
                        continue

                    positions.append({
                        "ticker": row["Ticker"],
                        "entry_price": entry_price,
                        "exit_date": next_day + pd.Timedelta(days=HOLD_DAYS),
                        "capital": capital
                    })

        equity += daily_pnl
        equity_curve.append(equity)

    equity_curve = pd.Series(equity_curve)

    if len(equity_curve) < 2:
        return None

    returns = equity_curve.pct_change().dropna()

    CAGR = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
    Sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
    MaxDD = (equity_curve / equity_curve.cummax() - 1).min()

    return CAGR, Sharpe, MaxDD

# =========================
# 実行
# =========================
results = []

for tp in TAKE_PROFIT_LIST:

    print(f"Running TAKE_PROFIT={tp}")

    res = run_backtest(tp)

    if res is None:
        continue

    CAGR, Sharpe, MaxDD = res

    results.append({
        "TAKE_PROFIT": tp,
        "CAGR": CAGR,
        "Sharpe": Sharpe,
        "MaxDD": MaxDD
    })

# =========================
# 結果
# =========================
result_df = pd.DataFrame(results)

print("\n=== TAKE PROFIT OPTIMIZATION ===")
print(result_df.sort_values("CAGR", ascending=False))
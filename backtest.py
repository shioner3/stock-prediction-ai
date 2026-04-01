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

THRESHOLD = 0.30
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
df["Year"] = df["Date"].dt.year

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
# 相場分類
# =========================
def add_regime(df):
    market = df.groupby("Date")["Return_1"].mean()
    ma20 = market.rolling(20).mean()

    regime_map = {}

    for d in market.index:
        val = ma20.loc[d]

        if pd.isna(val):
            regime_map[d] = "SIDE"
        elif val > 0.001:
            regime_map[d] = "UP"
        elif val < -0.001:
            regime_map[d] = "DOWN"
        else:
            regime_map[d] = "SIDE"

    df["regime"] = df["Date"].map(regime_map)
    return df

# =========================
# バックテスト
# =========================
def run_backtest(train_df, test_df):

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(train_df[FEATURES], train_df[TARGET])

    test_df = test_df.copy()
    test_df["pred"] = model.predict_proba(test_df[FEATURES])[:, 1]
    test_df = add_regime(test_df)

    equity = INITIAL_CAPITAL
    equity_curve = []
    regime_log = []

    positions = []

    dates = sorted(test_df["Date"].unique())
    date_index = {d: i for i, d in enumerate(dates)}

    for d in dates:

        today = test_df[test_df["Date"] == d]

        # 🔥 regime先に確定（これがポイント）
        regime = "SIDE"
        if not today.empty:
            regime = today["regime"].iloc[0]

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

                # 翌日チェック（continueしない）
                if d in date_index and date_index[d] + 1 < len(dates):

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

        # =========================
        # 日次更新（必ず実行）
        # =========================
        equity += daily_pnl
        equity_curve.append(equity)
        regime_log.append(regime)

    equity_curve = pd.Series(equity_curve)
    returns = equity_curve.pct_change().fillna(0)

    result = pd.DataFrame({
        "equity": equity_curve,
        "return": returns,
        "regime": regime_log
    })

    return result

# =========================
# ローリング
# =========================
results = []

years = sorted(df["Year"].unique())

for start in range(min(years), max(years) - 3):

    train = df[(df["Year"] >= start) & (df["Year"] < start+3)]
    test  = df[(df["Year"] >= start+3) & (df["Year"] < start+4)]

    if len(test) == 0:
        continue

    print(f"Running: {start}-{start+3} → {start+4}")

    res = run_backtest(train, test)
    res["window"] = f"{start}-{start+3}"

    results.append(res)

all_results = pd.concat(results)

# =========================
# 相場分析
# =========================
summary = all_results.groupby("regime")["return"].agg([
    ("mean_return", "mean"),
    ("vol", "std"),
    ("count", "count")
])

summary["Sharpe"] = summary["mean_return"] / (summary["vol"] + 1e-9) * np.sqrt(252)

print("\n=== REGIME ANALYSIS ===")
print(summary)

# =========================
# 全体
# =========================
equity = (1 + all_results["return"]).cumprod()

CAGR = equity.iloc[-1] ** (252 / len(equity)) - 1
Sharpe = all_results["return"].mean() / (all_results["return"].std() + 1e-9) * np.sqrt(252)
MaxDD = (equity / equity.cummax() - 1).min()

print("\n=== TOTAL RESULT (ROLLING) ===")
print("CAGR:", CAGR)
print("Sharpe:", Sharpe)
print("MaxDD:", MaxDD)
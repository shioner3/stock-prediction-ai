import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

FEATURES = [
    "Return_1_rank",
    "MA5_ratio_rank",
    "MA25_ratio_rank",
    "MA75_ratio_rank",
    "Volatility_rank",
    "Volume_change_rank",
    "HL_range_rank",
    "RSI_rank"
]

TARGET = "Target"

HOLD_DAYS = 5
STOP_LOSS = -0.03
INITIAL_CAPITAL = 1.0
TRAIN_INTERVAL = 20  # 月1

# =========================
# レジーム設定
# =========================
def get_regime(score):
    if score > 0.01:
        return "bull"
    elif score > 0:
        return "neutral"
    else:
        return "bear"

# 🔥 修正ポイント
REGIME_CONFIG = {
    "bull": {
        "quantile": 0.7,
        "max_positions": 5
    },
    "neutral": {
        "quantile": 0.85,   # 厳選
        "max_positions": 2  # 絞る
    },
    "bear": {
        "quantile": 1.0,   # 実質通さない
        "max_positions": 0 # 完全ノートレード
    }
}

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# =========================
# レジーム
# =========================
df["MarketRet"] = df.groupby("Date")["Close"].transform(
    lambda x: x.pct_change().mean()
).fillna(0)

df["MarketMA20"] = df["MarketRet"].rolling(20).mean()
df["RegimeScore"] = df["MarketMA20"]
df["Regime"] = df["RegimeScore"].apply(get_regime)

# =========================
# 年
# =========================
df["Year"] = df["Date"].dt.year
years = sorted(df["Year"].unique())

results = []

# =========================
# 🔁 ローリング
# =========================
for i in range(4, len(years) - 1):

    train_years = years[i-4:i]
    test_year = years[i]

    train_df = df[df["Year"].isin(train_years)]
    test_df = df[df["Year"] == test_year]

    print(f"\n===== {train_years} → {test_year} =====")

    equity = INITIAL_CAPITAL
    equity_curve = []

    model = None
    current_positions = []
    entry_index = None

    position_counts = []
    trade_count = 0

    dates = sorted(test_df["Date"].unique())

    for j in range(len(dates) - 1):

        d = dates[j]
        next_d = dates[j + 1]

        today = test_df[test_df["Date"] == d].copy()
        tomorrow = test_df[test_df["Date"] == next_d].copy()

        if len(today) == 0 or len(tomorrow) == 0:
            equity_curve.append(equity)
            position_counts.append(len(current_positions))
            continue

        # =========================
        # レジーム
        # =========================
        regime = today["Regime"].iloc[0]
        config = REGIME_CONFIG[regime]

        # =========================
        # 学習
        # =========================
        train_until = df[df["Date"] < d]

        if len(train_until) > 1000 and (model is None or j % TRAIN_INTERVAL == 0):
            model = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42
            )
            model.fit(train_until[FEATURES], train_until[TARGET])

        if model is None:
            equity_curve.append(equity)
            position_counts.append(len(current_positions))
            continue

        # =========================
        # ① エントリー（5日ごと）
        # =========================
        if j % HOLD_DAYS == 0:

            current_positions = []
            entry_index = j

            # 🔥 bearならスキップ
            if config["max_positions"] == 0:
                equity_curve.append(equity)
                position_counts.append(0)
                continue

            today["Pred"] = model.predict(today[FEATURES])

            th = today["Pred"].quantile(config["quantile"])
            today = today[today["Pred"] > th]

            if len(today) > 0:
                picks = today.sort_values("Pred", ascending=False).head(config["max_positions"])

                for _, row in picks.iterrows():

                    tomorrow_row = tomorrow[tomorrow["Ticker"] == row["Ticker"]]

                    if len(tomorrow_row) == 0:
                        continue

                    entry_price = tomorrow_row["Open"].values[0]

                    current_positions.append({
                        "ticker": row["Ticker"],
                        "entry_price": entry_price
                    })

                    trade_count += 1

        # =========================
        # ② 評価
        # =========================
        if len(current_positions) == 0:
            equity_curve.append(equity)
            position_counts.append(0)
            continue

        rets = []

        for pos in current_positions:

            current_row = today[today["Ticker"] == pos["ticker"]]

            if len(current_row) == 0:
                continue

            price = current_row["Close"].values[0]

            ret = (price - pos["entry_price"]) / pos["entry_price"]
            ret = max(ret, STOP_LOSS)

            rets.append(ret)

        if len(rets) == 0:
            equity_curve.append(equity)
            position_counts.append(len(current_positions))
            continue

        portfolio_ret = np.mean(rets)

        # =========================
        # ③ 決済
        # =========================
        if (j - entry_index + 1) == HOLD_DAYS:
            equity *= (1 + portfolio_ret)

        equity_curve.append(equity)
        position_counts.append(len(current_positions))

    # =========================
    # 評価
    # =========================
    equity_curve = pd.Series(equity_curve)
    returns = equity_curve.pct_change().dropna()

    if len(returns) == 0:
        continue

    cagr = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
    sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
    maxdd = (equity_curve / equity_curve.cummax() - 1).min()

    avg_pos = np.mean(position_counts)

    results.append({
        "train_period": f"{train_years[0]}-{train_years[-1]}",
        "test_period": f"{test_year}-{test_year+1}",
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": maxdd,
        "Avg_Positions": avg_pos,
        "Trades": trade_count
    })

# =========================
# 出力
# =========================
res_df = pd.DataFrame(results)

print("\n===== Rolling Backtest Result =====")
print(res_df)

print("\n===== Average =====")
print("Avg CAGR  :", res_df["CAGR"].mean())
print("Avg Sharpe:", res_df["Sharpe"].mean())
print("Avg MaxDD :", res_df["MaxDD"].mean())
print("Avg Pos   :", res_df["Avg_Positions"].mean())
print("Trades    :", res_df["Trades"].sum())

res_df.to_csv("rolling_backtest_realistic_v2.csv", index=False)
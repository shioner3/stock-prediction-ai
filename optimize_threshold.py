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

# レジーム別パラメータ
REGIME_CONFIG = {
    "bull": {
        "quantile": 0.7,
        "max_positions": 5
    },
    "neutral": {
        "quantile": 0.8,
        "max_positions": 3
    },
    "bear": {
        "quantile": 0.9,
        "max_positions": 1  # or 0で完全停止も可
    }
}

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# =========================
# 市場レジーム（リークなし）
# =========================
df["MarketRet"] = df.groupby("Date")["Close"].transform(
    lambda x: x.pct_change().mean()
).fillna(0)

df["MarketMA20"] = df["MarketRet"].rolling(20).mean()
df["RegimeScore"] = df["MarketMA20"]

df["Regime"] = df["RegimeScore"].apply(get_regime)

# =========================
# 年情報
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

    # =========================
    # テスト
    # =========================
    equity = INITIAL_CAPITAL
    equity_curve = []

    model = None
    current_positions = []
    entry_index = None

    dates = sorted(test_df["Date"].unique())

    for j in range(len(dates) - 1):  # ← 翌日使うので-1

        d = dates[j]
        next_d = dates[j + 1]

        today = test_df[test_df["Date"] == d].copy()
        tomorrow = test_df[test_df["Date"] == next_d].copy()

        if len(today) == 0 or len(tomorrow) == 0:
            equity_curve.append(equity)
            continue

        # =========================
        # レジーム取得
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
            continue

        # =========================
        # ① エントリー（5日ごと）
        # =========================
        if j % HOLD_DAYS == 0:

            current_positions = []
            entry_index = j

            today["Pred"] = model.predict(today[FEATURES])

            # 🔥 分位フィルター
            th = today["Pred"].quantile(config["quantile"])
            today = today[today["Pred"] > th]

            if len(today) > 0:
                picks = today.sort_values("Pred", ascending=False).head(config["max_positions"])

                for _, row in picks.iterrows():

                    # 🔥 翌日Openでエントリー
                    tomorrow_row = tomorrow[tomorrow["Ticker"] == row["Ticker"]]

                    if len(tomorrow_row) == 0:
                        continue

                    entry_price = tomorrow_row["Open"].values[0]

                    current_positions.append({
                        "ticker": row["Ticker"],
                        "entry_price": entry_price
                    })

        # =========================
        # ② ポジション評価
        # =========================
        if len(current_positions) == 0:
            equity_curve.append(equity)
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
            continue

        portfolio_ret = np.mean(rets)

        # =========================
        # ③ 決済（5日後）
        # =========================
        if (j - entry_index + 1) == HOLD_DAYS:
            equity *= (1 + portfolio_ret)

        equity_curve.append(equity)

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

    results.append({
        "train_period": f"{train_years[0]}-{train_years[-1]}",
        "test_period": f"{test_year}-{test_year+1}",
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": maxdd
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

res_df.to_csv("rolling_backtest_realistic.csv", index=False)
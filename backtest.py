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

HOLD_DAYS = 5
STOP_LOSS = -0.03
INITIAL_CAPITAL = 1.0
TRAIN_INTERVAL = 20

# weight制御
MAX_WEIGHT = 0.4

# =========================
# レジーム
# =========================
def get_regime(score, trend):
    if score > 0.003 and trend > 0:
        return "bull"
    elif score > -0.003:
        return "neutral"
    else:
        return "bear"


REGIME_CONFIG = {
    "bull": {"quantile": 0.6, "max_positions": 8},
    "neutral": {"quantile": 0.8, "max_positions": 4},
    "bear": {"quantile": 1.0, "max_positions": 1}
}

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

df["MarketRet"] = df.groupby("Date")["Close"].transform(lambda x: x.pct_change().mean()).fillna(0)
df["MarketMA20"] = df["MarketRet"].rolling(20).mean()
df["MarketMA60"] = df["MarketRet"].rolling(60).mean()

df["Regime"] = df.apply(
    lambda x: get_regime(x["MarketMA20"], x["MarketMA20"] - x["MarketMA60"]),
    axis=1
)

df["Year"] = df["Date"].dt.year
years = sorted(df["Year"].unique())

results = []

# =========================
# ローリング
# =========================
for i in range(4, len(years) - 1):

    train_years = years[i-4:i]
    test_year = years[i]

    test_df = df[df["Year"] == test_year]

    print(f"\n===== {train_years} → {test_year} =====")

    equity = INITIAL_CAPITAL
    equity_curve = []

    model = None

    positions = []  # 🔥 ポジション管理

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
            position_counts.append(len(positions))
            continue

        regime = today["Regime"].iloc[0]
        config = REGIME_CONFIG[regime]

        # =========================
        # 学習
        # =========================
        train_until = df[df["Date"] < d]

        if len(train_until) > 1000 and (model is None or j % TRAIN_INTERVAL == 0):
            model = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            )
            model.fit(train_until[FEATURES], train_until["Target"])

        if model is None:
            equity_curve.append(equity)
            position_counts.append(len(positions))
            continue

        # =========================
        # ① エントリー
        # =========================
        if j % HOLD_DAYS == 0:

            positions = []

            if config["max_positions"] > 0:

                today["pred"] = model.predict(today[FEATURES])

                th = today["pred"].quantile(config["quantile"])
                picks = today[today["pred"] > th].sort_values("pred", ascending=False)

                picks = picks.head(config["max_positions"]).copy()

                # =========================
                # weight設計（正規分布型）
                # =========================
                picks["vol"] = picks["Volatility_rank"] + 1e-6
                picks["raw_w"] = 1 / picks["vol"]

                picks["weight"] = picks["raw_w"] / picks["raw_w"].sum()
                picks["weight"] = picks["weight"].clip(0, MAX_WEIGHT)
                picks["weight"] = picks["weight"] / picks["weight"].sum()

                for _, row in picks.iterrows():

                    tmr = tomorrow[tomorrow["Ticker"] == row["Ticker"]]

                    if len(tmr) == 0:
                        continue

                    positions.append({
                        "ticker": row["Ticker"],
                        "entry_price": tmr["Open"].values[0],
                        "entry_day": j,
                        "weight": row["weight"],
                        "alive": True
                    })

                    trade_count += 1

        # =========================
        # ② ポジション管理
        # =========================
        new_positions = []

        for pos in positions:

            if not pos["alive"]:
                continue

            cur = today[today["Ticker"] == pos["ticker"]]

            if len(cur) == 0:
                new_positions.append(pos)
                continue

            price = cur["Close"].values[0]
            ret = (price - pos["entry_price"]) / pos["entry_price"]

            hold_days = j - pos["entry_day"] + 1

            # =========================
            # STOP
            # =========================
            if ret <= STOP_LOSS:
                exit_price = tomorrow[tomorrow["Ticker"] == pos["ticker"]]["Open"].values[0]
                pnl = (exit_price - pos["entry_price"]) / pos["entry_price"]

                equity *= (1 + pnl * pos["weight"])
                continue

            # =========================
            # 通常決済
            # =========================
            if hold_days >= HOLD_DAYS:
                equity *= (1 + ret * pos["weight"])
                continue

            new_positions.append(pos)

        positions = new_positions

        equity_curve.append(equity)
        position_counts.append(len(positions))

    # =========================
    # 評価
    # =========================
    equity_curve = pd.Series(equity_curve)
    returns = equity_curve.pct_change().dropna()

    cagr = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
    sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
    maxdd = (equity_curve / equity_curve.cummax() - 1).min()

    results.append({
        "train": f"{train_years[0]}-{train_years[-1]}",
        "test": f"{test_year}-{test_year+1}",
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": maxdd,
        "AvgPos": np.mean(position_counts),
        "Trades": trade_count
    })

# =========================
# 出力
# =========================
res = pd.DataFrame(results)

print(res)
print("\nAVG CAGR:", res["CAGR"].mean())
print("AVG Sharpe:", res["Sharpe"].mean())
print("AVG DD:", res["MaxDD"].mean())
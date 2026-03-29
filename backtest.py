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
TRAIN_INTERVAL = 20

# 🔥 weight制御
MAX_WEIGHT = 0.4
MIN_WEIGHT = 0.05

# =========================
# レジーム（強化版）
# =========================
def get_regime(score, trend):

    if score > 0.003 and trend > 0:
        return "bull"

    elif score > -0.003:
        return "neutral"

    else:
        return "bear"


REGIME_CONFIG = {
    "bull": {"quantile": 0.7, "max_positions": 5},
    "neutral": {"quantile": 0.85, "max_positions": 3},
    "bear": {"quantile": 1.0, "max_positions": 0}
}

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# =========================
# 市場レジーム（強化）
# =========================
df["MarketRet"] = df.groupby("Date")["Close"].transform(
    lambda x: x.pct_change().mean()
).fillna(0)

df["MarketMA20"] = df["MarketRet"].rolling(20).mean()
df["MarketMA60"] = df["MarketRet"].rolling(60).mean()

df["RegimeScore"] = df["MarketMA20"]
df["Trend"] = df["MarketMA20"] - df["MarketMA60"]

df["Regime"] = df.apply(lambda x: get_regime(x["RegimeScore"], x["Trend"]), axis=1)

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

    test_df = df[df["Year"] == test_year]

    print(f"\n===== {train_years} → {test_year} =====")

    equity = INITIAL_CAPITAL
    equity_curve = []

    model = None
    current_positions = []

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
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            )
            model.fit(train_until[FEATURES], train_until[TARGET])

        if model is None:
            equity_curve.append(equity)
            position_counts.append(len(current_positions))
            continue

        # =========================
        # ① エントリー
        # =========================
        if j % HOLD_DAYS == 0:

            current_positions = []

            if config["max_positions"] > 0:

                today["Pred"] = model.predict(today[FEATURES])

                # 🔥 ボラ取得
                today["Volatility"] = today["Volatility_rank"] + 1e-6

                # 🔥 weight = 1/vol
                today["raw_weight"] = 1 / today["Volatility"]

                # 🔥 正規化
                today["weight"] = today["raw_weight"] / today["raw_weight"].sum()

                # 🔥 上限・下限クリップ
                today["weight"] = today["weight"].clip(MIN_WEIGHT, MAX_WEIGHT)

                # 🔥 再正規化
                today["weight"] = today["weight"] / today["weight"].sum()

                # 上位抽出
                th = today["Pred"].quantile(config["quantile"])
                picks = today[today["Pred"] > th].sort_values("Pred", ascending=False).head(config["max_positions"])

                for _, row in picks.iterrows():

                    tomorrow_row = tomorrow[tomorrow["Ticker"] == row["Ticker"]]

                    if len(tomorrow_row) == 0:
                        continue

                    entry_price = tomorrow_row["Open"].values[0]

                    current_positions.append({
                        "ticker": row["Ticker"],
                        "entry_price": entry_price,
                        "entry_day": j,
                        "weight": row["weight"],
                        "stop_flag": False
                    })

                    trade_count += 1

        # =========================
        # ② ポジション評価
        # =========================
        new_positions = []

        for pos in current_positions:

            # STOP確定
            if pos["stop_flag"]:
                tomorrow_row = tomorrow[tomorrow["Ticker"] == pos["ticker"]]

                if len(tomorrow_row) > 0:
                    exit_price = tomorrow_row["Open"].values[0]
                    ret = (exit_price - pos["entry_price"]) / pos["entry_price"]

                    equity *= (1 + ret * pos["weight"])

                continue

            current_row = today[today["Ticker"] == pos["ticker"]]

            if len(current_row) == 0:
                new_positions.append(pos)
                continue

            price = current_row["Close"].values[0]
            ret = (price - pos["entry_price"]) / pos["entry_price"]

            # STOP判定
            if ret <= STOP_LOSS:
                pos["stop_flag"] = True
                new_positions.append(pos)
                continue

            # HOLD満了
            hold_days = j - pos["entry_day"] + 1

            if hold_days >= HOLD_DAYS:
                equity *= (1 + ret * pos["weight"])
                continue

            new_positions.append(pos)

        current_positions = new_positions

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

res_df.to_csv("rolling_backtest_realistic_v6.csv", index=False)
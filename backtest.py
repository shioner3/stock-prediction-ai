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

TARGET = "FutureReturn_5"

HOLD_DAYS = 5
STOP_LOSS = -0.03
INITIAL_CAPITAL = 1.0
TRAIN_INTERVAL = 20
MAX_WEIGHT = 0.4

# 🔥 集中投資＋調整
REGIME_CONFIG = {
    "bull": {"quantile": 0.65, "max_positions": 4},
    "neutral": {"quantile": 0.8, "max_positions": 4},
    "bear": {"quantile": 1.0, "max_positions": 0}
}

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

# =========================
# 市場レジーム
# =========================
df["MarketRet"] = df.groupby("Date")["Close"].transform("mean").pct_change().fillna(0)

market = df.groupby("Date")["MarketRet"].mean().sort_index()
market_ma20 = market.rolling(20).mean()
market_ma60 = market.rolling(60).mean()

market_df = pd.DataFrame({
    "MarketMA20": market_ma20,
    "MarketMA60": market_ma60
}).fillna(0)

df = df.merge(market_df, left_on="Date", right_index=True, how="left")

df["Regime"] = np.where(
    (df["MarketMA20"] > 0.003) & (df["MarketMA20"] - df["MarketMA60"] > 0),
    "bull",
    np.where(df["MarketMA20"] > -0.003, "neutral", "bear")
)

# =========================
# 年
# =========================
df["Year"] = df["Date"].dt.year
years = sorted(df["Year"].unique())

results = []

# =========================
# バックテスト
# =========================
for i in range(4, len(years) - 1):

    train_years = years[i-4:i]
    test_year = years[i]

    test_df = df[df["Year"] == test_year].copy()

    print(f"\n===== {train_years} → {test_year} =====")

    equity = INITIAL_CAPITAL
    equity_curve = []

    model = None
    positions = []

    position_counts = []
    trade_count = 0
    trade_returns = []

    dates = sorted(test_df["Date"].unique())

    for j in range(len(dates) - 1):

        d = dates[j]
        next_d = dates[j + 1]

        today = test_df[test_df["Date"] == d].copy()
        tomorrow = test_df[test_df["Date"] == next_d].copy()

        if today.empty:
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
            model.fit(train_until[FEATURES], train_until[TARGET])

        if model is None:
            equity_curve.append(equity)
            position_counts.append(len(positions))
            continue

        # =========================
        # エントリー
        # =========================
        if j % HOLD_DAYS == 0:

            positions = []

            if config["max_positions"] > 0:

                today_pred = today.copy()
                today_pred["pred"] = model.predict(today_pred[FEATURES])

                # 🔥 フィルタ強化（軽め）
                today_pred = today_pred[today_pred["pred"] > 0.005]

                if today_pred.empty:
                    equity_curve.append(equity)
                    position_counts.append(0)
                    continue

                # 🔥 スコア合成
                today_pred["score"] = (
                    today_pred["pred"] * today_pred["MA5_ratio_rank"]
                )

                th = today_pred["score"].quantile(config["quantile"])
                picks = today_pred[today_pred["score"] > th].copy()

                picks = picks.sort_values("score", ascending=False)
                picks = picks.head(config["max_positions"])

                if picks.empty:
                    equity_curve.append(equity)
                    position_counts.append(0)
                    continue

                # =========================
                # 🔥 安定weight（ボラ逆数）
                # =========================
                picks["vol"] = picks["Volatility_rank"] + 1e-6
                picks["weight"] = 1 / np.sqrt(picks["vol"])

                picks["weight"] /= picks["weight"].sum()
                picks["weight"] = picks["weight"].clip(0, MAX_WEIGHT)
                picks["weight"] /= picks["weight"].sum()

                for _, row in picks.iterrows():

                    tmr = tomorrow[tomorrow["Ticker"] == row["Ticker"]]

                    if tmr.empty:
                        continue

                    positions.append({
                        "ticker": row["Ticker"],
                        "entry_price": tmr["Open"].iloc[0],
                        "entry_day": j,
                        "weight": row["weight"]
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

            hold_days = j - pos["entry_day"] + 1

            # =========================
            # STOP
            # =========================
            if ret <= STOP_LOSS:

                tmr = tomorrow[tomorrow["Ticker"] == pos["ticker"]]

                if not tmr.empty:
                    exit_price = tmr["Open"].iloc[0]
                else:
                    exit_price = price

                pnl = (exit_price - pos["entry_price"]) / pos["entry_price"]

                equity *= (1 + pnl * pos["weight"])
                trade_returns.append(pnl)

                continue

            # =========================
            # 通常決済
            # =========================
            if hold_days >= HOLD_DAYS:

                pnl = ret
                equity *= (1 + pnl * pos["weight"])
                trade_returns.append(pnl)

                continue

            new_positions.append(pos)

        positions = new_positions

        equity_curve.append(equity)
        position_counts.append(len(positions))

    # =========================
    # 評価
    # =========================
    equity_curve = pd.Series(equity_curve)

    if len(equity_curve) < 2:
        continue

    returns = equity_curve.pct_change().dropna()

    cagr = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
    sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
    maxdd = (equity_curve / equity_curve.cummax() - 1).min()

    trade_returns = np.array(trade_returns)

    if len(trade_returns) > 0:
        win_rate = (trade_returns > 0).mean()
        avg_win = trade_returns[trade_returns > 0].mean() if np.any(trade_returns > 0) else 0
        avg_loss = trade_returns[trade_returns <= 0].mean() if np.any(trade_returns <= 0) else 0
        pf = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
    else:
        win_rate = avg_win = avg_loss = pf = np.nan

    results.append({
        "train": f"{train_years[0]}-{train_years[-1]}",
        "test": f"{test_year}-{test_year+1}",
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": maxdd,
        "WinRate": win_rate,
        "AvgWin": avg_win,
        "AvgLoss": avg_loss,
        "PF": pf,
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
print("AVG WinRate:", res["WinRate"].mean())
print("AVG PF:", res["PF"].mean())
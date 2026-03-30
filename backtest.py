import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

FEATURES = [
    "Return_1_rank",
    "Return_3_rank",
    "MA3_ratio_rank",
    "MA5_ratio_rank",
    "MA10_ratio_rank",
    "Volatility_rank",
    "Volume_change_rank",
    "Volume_ratio_rank",
    "HL_range_rank",
    "RSI_rank"
]

TARGET = "Target"

STOP_LOSS = -0.03
INITIAL_CAPITAL = 1.0
MAX_WEIGHT = 0.4

# 🔥 緩和
PRED_THRESHOLD = 0.51
MARKET_THRESHOLD = 0.48

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

# =========================
# 市場レジーム
# =========================
df["MarketRet"] = df.groupby("Date")["Close"].transform("mean").pct_change().fillna(0)

market = df.groupby("Date")["MarketRet"].mean().sort_index()
market_ma20 = market.rolling(20).mean()

df = df.merge(
    pd.DataFrame({"MarketMA20": market_ma20}).fillna(0),
    left_on="Date",
    right_index=True,
    how="left"
)

df["Regime"] = np.where(
    df["MarketMA20"] > 0.002, "bull",
    np.where(df["MarketMA20"] > -0.002, "neutral", "bear")
)

# =========================
# 年分割
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

    trade_count = 0
    trade_returns = []

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

        regime = today["Regime"].iloc[0]

        # =========================
        # 月次学習
        # =========================
        current_month = d.month

        if model is None or current_month != prev_month:

            train_until = df[df["Date"] < d]

            if len(train_until) > 2000:
                model = LGBMRegressor(
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
        # 予測
        # =========================
        today_pred = today.copy()
        today_pred["pred"] = model.predict(today_pred[FEATURES])

        # =========================
        # 地雷フィルター（軽め）
        # =========================
        if "is_earnings" in today_pred.columns:
            today_pred = today_pred[today_pred["is_earnings"] == 0]

        if "limit_up_flag" in today_pred.columns:
            today_pred = today_pred[today_pred["limit_up_flag"] == 0]

        if today_pred.empty:
            equity_curve.append(equity)
            continue

        # =========================
        # 市場フィルター（緩和）
        # =========================
        market_score = today_pred["pred"].mean()

        if np.isnan(market_score):
            market_score = 0

        # 🔥 完全停止しない
        if market_score < MARKET_THRESHOLD:
            max_positions = 2
        else:
            max_positions = 5 if regime == "bull" else 3

        # =========================
        # スコアフィルター
        # =========================
        candidates = today_pred[today_pred["pred"] > PRED_THRESHOLD]

        # 🔥 fallback（必ずトレード）
        if len(candidates) < 3:
            candidates = today_pred.sort_values("pred", ascending=False).head(5)

        # =========================
        # エントリー（常時）
        # =========================
        if len(positions) == 0:

            picks = candidates.sort_values("pred", ascending=False).head(max_positions)

            # weight
            picks["vol"] = picks["Volatility_rank"] + 1e-6
            picks["weight"] = 1 / np.sqrt(picks["vol"])
            picks["weight"] /= picks["weight"].sum()

            hold_days = 3 if regime != "bull" else 4

            for _, row in picks.iterrows():

                tmr = tomorrow[tomorrow["Ticker"] == row["Ticker"]]
                if tmr.empty:
                    continue

                positions.append({
                    "ticker": row["Ticker"],
                    "entry_price": tmr["Open"].iloc[0],
                    "entry_day": j,
                    "weight": row["weight"],
                    "hold_days": hold_days
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

            # 損切り
            if ret <= STOP_LOSS:
                tmr = tomorrow[tomorrow["Ticker"] == pos["ticker"]]
                exit_price = tmr["Open"].iloc[0] if not tmr.empty else price

                pnl = (exit_price - pos["entry_price"]) / pos["entry_price"]
                equity *= (1 + pnl * pos["weight"])
                trade_returns.append(pnl)
                continue

            # 利確
            if hold_days_now >= pos["hold_days"]:
                pnl = ret
                equity *= (1 + pnl * pos["weight"])
                trade_returns.append(pnl)
                continue

            new_positions.append(pos)

        positions = new_positions
        equity_curve.append(equity)

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

    win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0

    pf = np.nan
    if len(trade_returns) > 0:
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns <= 0]
        if len(wins) > 0 and len(losses) > 0:
            pf = wins.mean() / abs(losses.mean())

    results.append({
        "train": f"{train_years[0]}-{train_years[-1]}",
        "test": f"{test_year}",
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": maxdd,
        "WinRate": win_rate,
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
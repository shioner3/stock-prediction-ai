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
STOP_LOSS = -0.05
INITIAL_CAPITAL = 1.0
TRAIN_INTERVAL = 20
MAX_WEIGHT = 0.4

TEST_YEAR = 2026   # 🔥 ここが今回の主役

REGIME_CONFIG = {
    "bull": {"quantile": 0.6, "max_positions": 8},
    "neutral": {"quantile": 0.8, "max_positions": 4},
    "bear": {"quantile": 1.0, "max_positions": 1}
}

# =========================
# データ読み込み
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
# 学習期間 / テスト期間
# =========================
train_df = df[df["Date"] < f"{TEST_YEAR}-01-01"]
test_df  = df[df["Date"].dt.year == TEST_YEAR].copy()

print(f"\n===== TRAIN < {TEST_YEAR} | TEST = {TEST_YEAR} =====")

# =========================
# 初期化
# =========================
equity = INITIAL_CAPITAL
equity_curve = []

model = None
positions = []

position_counts = []
trade_count = 0
trade_returns = []

dates = sorted(test_df["Date"].unique())

# =========================
# バックテスト
# =========================
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
    # モデル更新（未来データ禁止）
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
    # エントリー
    # =========================
    if j % HOLD_DAYS == 0:

        positions = []

        if config["max_positions"] > 0:

            today["pred"] = model.predict(today[FEATURES])

            th = today["pred"].quantile(config["quantile"])
            picks = today[today["pred"] > th].copy()

            picks = picks.sort_values("pred", ascending=False)
            picks = picks.head(config["max_positions"])

            # =========================
            # weight（マイルド版）
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
        # STOP（現実的）
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
returns = equity_curve.pct_change().dropna()

cagr = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
maxdd = (equity_curve / equity_curve.cummax() - 1).min()

# =========================
# エッジ分解
# =========================
trade_returns = np.array(trade_returns)

win_rate = (trade_returns > 0).mean()
avg_win = trade_returns[trade_returns > 0].mean() if np.any(trade_returns > 0) else 0
avg_loss = trade_returns[trade_returns <= 0].mean() if np.any(trade_returns <= 0) else 0
pf = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

# =========================
# 出力
# =========================
print("\n===== 2026 BACKTEST =====")
print("CAGR     :", cagr)
print("Sharpe   :", sharpe)
print("MaxDD    :", maxdd)
print("WinRate  :", win_rate)
print("AvgWin   :", avg_win)
print("AvgLoss  :", avg_loss)
print("PF       :", pf)
print("Trades   :", trade_count)
print("AvgPos   :", np.mean(position_counts))
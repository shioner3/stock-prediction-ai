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

MAX_POSITIONS = 5
HOLD_DAYS = 5
STOP_LOSS = -0.03
INITIAL_CAPITAL = 1.0
TRAIN_INTERVAL = 20  # 月1学習
COST = 0.002

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

dates = sorted(df["Date"].unique())

# =========================
# 状態
# =========================
equity_curve = []
capital = INITIAL_CAPITAL

model = None

# 現在ポジション（5日固定）
current_positions = []
entry_index = None

# =========================
# 日次ループ
# =========================
for i, d in enumerate(dates):

    today = df[df["Date"] == d].copy()

    if len(today) == 0:
        equity_curve.append(capital)
        continue

    # =========================
    # 月1学習
    # =========================
    train_data = df[df["Date"] < d]

    if len(train_data) > 1000 and (model is None or i % TRAIN_INTERVAL == 0):
        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        )
        model.fit(train_data[FEATURES], train_data[TARGET])

    if model is None:
        equity_curve.append(capital)
        continue

    # =========================
    # ① 新規エントリー（5日ごと）
    # =========================
    if i % HOLD_DAYS == 0:

        # 前ポジションは必ず空になっている想定
        current_positions = []
        entry_index = i

        today["Pred"] = model.predict(today[FEATURES])

        picks = today.sort_values("Pred", ascending=False).head(MAX_POSITIONS)

        for _, row in picks.iterrows():
            current_positions.append({
                "ticker": row["Ticker"],
                "entry_price": row["Close"]
            })

    # =========================
    # ② ポジション評価
    # =========================
    if len(current_positions) == 0:
        equity_curve.append(capital)
        continue

    rets = []

    for pos in current_positions:

        current_row = today[today["Ticker"] == pos["ticker"]]

        if len(current_row) == 0:
            continue

        current_price = current_row["Close"].values[0]

        ret = (current_price - pos["entry_price"]) / pos["entry_price"]

        # STOP LOSS（含み）
        ret = max(ret, STOP_LOSS)

        rets.append(ret)

    if len(rets) == 0:
        equity_curve.append(capital)
        continue

    portfolio_ret = np.mean(rets)

    # =========================
    # ③ 5日後に確定
    # =========================
    if (i - entry_index + 1) == HOLD_DAYS:
        portfolio_ret -= COST * 2  # 往復コスト
        capital *= (1 + portfolio_ret)

    equity_curve.append(capital)

# =========================
# 結果
# =========================
equity_curve = pd.Series(equity_curve)
returns = equity_curve.pct_change().dropna()

days = len(equity_curve)

cagr = equity_curve.iloc[-1] ** (252 / days) - 1
sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
maxdd = (equity_curve / equity_curve.cummax() - 1).min()

print("\n========================")
print("5D FIXED HOLD BACKTEST")
print("========================")
print("CAGR:", cagr)
print("Sharpe:", sharpe)
print("MaxDD:", maxdd)
print("Days:", days)

pd.DataFrame({
    "equity": equity_curve
}).to_csv("backtest_5d_fixed.csv", index=False)
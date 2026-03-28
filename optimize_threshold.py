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

TOP_K = 5
HOLD_DAYS = 5
STOP_LOSS = -0.03
INITIAL_CAPITAL = 1.0

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

dates = sorted(df["Date"].unique())

# =========================
# ポジション管理
# =========================
positions = []
equity_curve = []
capital = INITIAL_CAPITAL

model = None

# =========================
# 日次ループ
# =========================
for i, d in enumerate(dates):

    today = df[df["Date"] == d].copy()

    if len(today) == 0:
        equity_curve.append(capital)
        continue

    # =========================
    # 学習
    # =========================
    train_data = df[df["Date"] < d]

    if len(train_data) > 1000 and (model is None or i % 20 == 0):
        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        )
        model.fit(train_data[FEATURES], train_data[TARGET])

    if model is None:
        equity_curve.append(capital)
        continue

    # =========================
    # ① エントリー（TOP5）
    # =========================
    today["Pred"] = model.predict(today[FEATURES])
    picks = today.sort_values("Pred", ascending=False).head(TOP_K)

    for _, row in picks.iterrows():
        positions.append({
            "ticker": row["Ticker"],
            "entry_price": row["Close"],
            "entry_index": i,
            "exit_index": i + HOLD_DAYS
        })

    # =========================
    # ② ポジション評価（mark-to-market）
    # =========================
    daily_realized = 0.0
    new_positions = []

    for pos in positions:

        current_row = today[today["Ticker"] == pos["ticker"]]

        if len(current_row) == 0:
            continue

        current_price = current_row["Close"].values[0]

        ret = (current_price - pos["entry_price"]) / pos["entry_price"]

        # STOP LOSS（含み損ベース）
        ret = max(ret, STOP_LOSS)

        # ③ 5日経過で確定
        if i >= pos["exit_index"]:
            daily_realized += ret
        else:
            new_positions.append(pos)

    positions = new_positions

    # =========================
    # 資産更新（確定損益のみ）
    # =========================
    capital *= (1 + daily_realized)
    equity_curve.append(capital)

# =========================
# 結果
# =========================
equity_curve = pd.Series(equity_curve)
returns = equity_curve.pct_change().dropna()

cagr = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
maxdd = (equity_curve / equity_curve.cummax() - 1).min()

print("\n========================")
print("5D HOLD BACKTEST (CORRECTED)")
print("========================")
print("CAGR:", cagr)
print("Sharpe:", sharpe)
print("MaxDD:", maxdd)
print("Days:", len(equity_curve))

pd.DataFrame({"equity": equity_curve}).to_csv("backtest_equity.csv", index=False)
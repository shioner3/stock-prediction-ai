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
THRESHOLD = 0.03
STOP_LOSS = -0.03
INITIAL_CAPITAL = 1.0

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])
df = df.dropna(subset=["Date"])
df = df.sort_values("Date")

df["Year"] = df["Date"].dt.year
years = sorted(df["Year"].unique())

# =========================
# レジーム（リークなし）
# =========================
df["MarketRet"] = df.groupby("Ticker")["Close"].pct_change()
df["MarketRet"] = df["MarketRet"].fillna(0)

df["MarketMA20"] = df.groupby("Ticker")["MarketRet"].transform(
    lambda x: x.rolling(20).mean()
)

df["Regime"] = df["MarketMA20"] > 0

# =========================
# 日次資産曲線
# =========================
equity_curve = []
capital = INITIAL_CAPITAL

# =========================
# ローリング
# =========================
for i in range(3, len(years) - 1):

    train_years = years[i-3:i]
    test_year = years[i]

    train = df[df["Year"].isin(train_years)]
    test = df[df["Year"] == test_year]

    dates = sorted(test["Date"].unique())

    model = None
    last_month = None

    for d in dates:

        train_until_now = train[train["Date"] < d]

        if len(train_until_now) < 1000:
            continue

        # =========================
        # 月次学習
        # =========================
        if model is None or (d.year, d.month) != last_month:
            model = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42
            )
            model.fit(train_until_now[FEATURES], train_until_now[TARGET])
            last_month = (d.year, d.month)

        today = test[test["Date"] == d].copy()

        if len(today) == 0:
            continue

        # =========================
        # レジーム
        # =========================
        if not today["Regime"].iloc[0]:
            equity_curve.append(capital)
            continue

        # =========================
        # 予測
        # =========================
        today["Pred"] = model.predict(today[FEATURES])

        today = today[today["Pred"] > THRESHOLD]

        if len(today) == 0:
            equity_curve.append(capital)
            continue

        picks = today.sort_values("Pred", ascending=False).head(TOP_K)

        rets = picks["FutureReturn_5"].values

        # STOP LOSS
        rets = np.clip(rets, STOP_LOSS, None)

        # =========================
        # ポートフォリオリターン（超重要）
        # =========================
        daily_ret = np.mean(rets)

        capital *= (1 + daily_ret)

        equity_curve.append(capital)

# =========================
# 結果計算
# =========================
equity_curve = pd.Series(equity_curve)

returns = equity_curve.pct_change().dropna()

days = len(equity_curve)

cagr = equity_curve.iloc[-1] ** (252 / days) - 1
sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
maxdd = (equity_curve / equity_curve.cummax() - 1).min()

# =========================
# 出力
# =========================
print("\n========================")
print("CORRECT BACKTEST RESULT")
print("========================")
print("CAGR:", cagr)
print("Sharpe:", sharpe)
print("MaxDD:", maxdd)
print("Days:", days)

pd.DataFrame({
    "equity": equity_curve
}).to_csv("backtest_equity.csv", index=False)
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

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

# =========================
# 固定パラメータ
# =========================
THRESHOLD = 0.03
STOP_LOSS = -0.03

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values("Date")

df["Year"] = df["Date"].dt.year
years = sorted(df["Year"].unique())

# =========================
# 市場レジーム（超重要）
# ※ここでは全体平均リターンで簡易定義
# =========================
df["MarketRet"] = df.groupby("Date")["Target"].transform("mean")
df["MarketMA20"] = df["MarketRet"].rolling(20).mean()
df["Regime"] = df["MarketMA20"] > 0


all_returns = []

# =========================
# ローリング検証
# =========================
for i in range(3, len(years) - 1):

    train_years = years[i-3:i]
    test_year = years[i]

    print(f"\n===== {train_years} → {test_year} =====")

    train = df[df["Year"].isin(train_years)]
    test_df = df[df["Year"] == test_year]

    dates = sorted(test_df["Date"].unique())

    model = None
    last_train_month = None

    for d in dates:

        train_until_now = train[train["Date"] < d]

        if len(train_until_now) < 1000:
            continue

        # =========================
        # 月1学習
        # =========================
        current_month = (d.year, d.month)

        if model is None or current_month != last_train_month:
            model = LGBMRegressor()
            model.fit(train_until_now[FEATURES], train_until_now[TARGET])
            last_train_month = current_month

        # =========================
        # 当日データ
        # =========================
        today = test_df[test_df["Date"] == d].copy()

        if len(today) == 0:
            continue

        # =========================
        # 市場レジームフィルター
        # =========================
        regime_ok = df[df["Date"] == d]["Regime"].iloc[0]

        if not regime_ok:
            continue

        # =========================
        # 予測
        # =========================
        today["Pred"] = model.predict(today[FEATURES])

        # =========================
        # threshold
        # =========================
        today = today[today["Pred"] > THRESHOLD]

        if len(today) == 0:
            continue

        # =========================
        # TOP K
        # =========================
        picks = today.sort_values("Pred", ascending=False).head(TOP_K)

        # =========================
        # 損切り -3%
        # =========================
        ret = picks["FutureReturn_5"].mean()

        if ret < STOP_LOSS:
            ret = STOP_LOSS

        all_returns.append(ret)

# =========================
# 集計
# =========================
res = pd.Series(all_returns)

if len(res) == 0:
    print("No trades")
    exit()

cum = (1 + res).cumprod()

cagr = cum.iloc[-1] ** (252 / len(res)) - 1
sharpe = res.mean() / res.std() * np.sqrt(252)
maxdd = (cum / cum.cummax() - 1).min()

print("\n========================")
print("RESULT (Regime + Stoploss)")
print("========================")

print("CAGR:", cagr)
print("Sharpe:", sharpe)
print("MaxDD:", maxdd)
print("Trades:", len(res))

# 保存
pd.DataFrame({
    "return": res
}).to_csv("backtest_regime_stoploss.csv", index=False)
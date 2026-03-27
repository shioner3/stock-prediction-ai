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

TOP_K = 10
HOLD_DAYS = 5


# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

# 年取得
df["Year"] = df["Date"].dt.year
years = sorted(df["Year"].unique())


results_summary = []


# =========================
# 年ごとローリング
# =========================
for i in range(3, len(years) - 1):

    train_years = years[i-3:i]   # 3年学習
    test_year = years[i]         # 1年テスト

    print(f"\n===== {train_years} → {test_year} =====")

    train = df[df["Year"].isin(train_years)]
    test_df = df[df["Year"] == test_year]

    dates = sorted(test_df["Date"].unique())

    daily_returns = []

    for d in dates:

        train_until_now = train[train["Date"] < d]

        if len(train_until_now) < 1000:
            continue

        model = LGBMRegressor()
        model.fit(train_until_now[FEATURES], train_until_now[TARGET])

        today = test_df[test_df["Date"] == d].copy()

        if len(today) == 0:
            continue

        today["Pred"] = model.predict(today[FEATURES])

        picks = today.sort_values("Pred", ascending=False).head(TOP_K)

        ret = picks["FutureReturn_5"].mean()

        daily_returns.append(ret)

    if len(daily_returns) == 0:
        continue

    res = pd.Series(daily_returns)

    cum = (1 + res).cumprod()

    cagr = cum.iloc[-1] ** (252/len(res)) - 1
    sharpe = res.mean() / res.std() * np.sqrt(252)
    maxdd = (cum / cum.cummax() - 1).min()

    print("CAGR:", cagr)
    print("Sharpe:", sharpe)
    print("MaxDD:", maxdd)

    results_summary.append({
        "train_period": f"{train_years[0]}-{train_years[-1]}",
        "test_period": f"{test_year}-{test_year+1}",
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": maxdd
    })


# =========================
# まとめ
# =========================
summary_df = pd.DataFrame(results_summary)

print("\n===== Summary =====")
print(summary_df)

print("\n===== Average =====")
print("Avg CAGR   :", summary_df["CAGR"].mean())
print("Avg Sharpe :", summary_df["Sharpe"].mean())
print("Avg MaxDD  :", summary_df["MaxDD"].mean())

# 保存
summary_df.to_csv("backtest_summary.csv", index=False)
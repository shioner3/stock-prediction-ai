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

TOP_K = 5                 # 🔥 最大5銘柄
PRED_THRESHOLD = 0.0      # 🔥 期待値フィルター
HOLD_DAYS = 5


# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values("Date")

df["Year"] = df["Date"].dt.year
years = sorted(df["Year"].unique())

results_summary = []


# =========================
# ローリング
# =========================
for i in range(3, len(years) - 1):

    train_years = years[i-3:i]
    test_year = years[i]

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

        # =========================
        # 予測
        # =========================
        today["Pred"] = model.predict(today[FEATURES])

        # 🔥 フィルター（超重要）
        today = today[today["Pred"] > PRED_THRESHOLD]

        if len(today) == 0:
            continue  # ノートレード

        # 🔥 最大5銘柄
        picks = today.sort_values("Pred", ascending=False).head(TOP_K)

        # 🔥 外れ値カット（現実寄せ）
        ret = picks["FutureReturn_5"].clip(-0.3, 0.5).mean()

        daily_returns.append(ret)

    if len(daily_returns) == 0:
        continue

    res = pd.Series(daily_returns)

    cum = (1 + res).cumprod()

    cagr = cum.iloc[-1] ** (252 / len(res)) - 1
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
        "MaxDD": maxdd,
        "trades": len(res)
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
print("Avg Trades :", summary_df["trades"].mean())

summary_df.to_csv("backtest_summary.csv", index=False)
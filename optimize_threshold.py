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

# 🔥 最適化するthreshold候補
THRESHOLDS = [-0.01, 0.0, 0.005, 0.01, 0.02, 0.03]


# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values("Date")

df["Year"] = df["Date"].dt.year
years = sorted(df["Year"].unique())


threshold_results = []


# =========================
# thresholdごとに検証
# =========================
for TH in THRESHOLDS:

    print("\n========================")
    print(f"THRESHOLD: {TH}")
    print("========================")

    all_returns = []

    # =========================
    # 年ごとローリング
    # =========================
    for i in range(3, len(years) - 1):

        train_years = years[i-3:i]
        test_year = years[i]

        print(f"\n===== {train_years} → {test_year} =====")

        train = df[df["Year"].isin(train_years)]
        test_df = df[df["Year"] == test_year]

        dates = sorted(test_df["Date"].unique())

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

            # 🔥 threshold適用
            today = today[today["Pred"] > TH]

            if len(today) == 0:
                continue

            # 🔥 最大5銘柄
            picks = today.sort_values("Pred", ascending=False).head(TOP_K)

            # 🔥 外れ値カット
            ret = picks["FutureReturn_5"].clip(-0.3, 0.5).mean()

            all_returns.append(ret)

    if len(all_returns) == 0:
        continue

    res = pd.Series(all_returns)

    cum = (1 + res).cumprod()

    cagr = cum.iloc[-1] ** (252 / len(res)) - 1
    sharpe = res.mean() / res.std() * np.sqrt(252)
    maxdd = (cum / cum.cummax() - 1).min()

    print("\n--- RESULT ---")
    print("CAGR:", cagr)
    print("Sharpe:", sharpe)
    print("MaxDD:", maxdd)
    print("Trades:", len(res))

    threshold_results.append({
        "threshold": TH,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": maxdd,
        "trades": len(res)
    })


# =========================
# 最終結果
# =========================
result_df = pd.DataFrame(threshold_results)

print("\n========================")
print("最適化結果")
print("========================")

print(result_df.sort_values("Sharpe", ascending=False))

result_df.to_csv("threshold_optimization.csv", index=False)
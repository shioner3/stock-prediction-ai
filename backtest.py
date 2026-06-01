import pandas as pd
import numpy as np

# =========================
# ファイル
# =========================
INPUT_FILE = "stock_data/accumulation_score.parquet"

# =========================
# 設定
# =========================
TOP_N = 20

HOLD_DAYS_LIST = [
    20,
    60,
    120
]

# =========================
# 読み込み
# =========================
print("Loading...")

df = pd.read_parquet(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)

# =========================
# Future Return
# =========================
for hold_days in HOLD_DAYS_LIST:

    df[f"FutureRet_{hold_days}"] = (
        df.groupby("Ticker")["Close"]
        .shift(-hold_days)
        / df["Close"]
        - 1
    )

# =========================
# バックテスト
# =========================
results = []

dates = sorted(
    df["Date"].unique()
)

for date in dates:

    daily = df[
        df["Date"] == date
    ].copy()

    if len(daily) < TOP_N:
        continue

    daily = daily.dropna(
        subset=["AccumulationScore"]
    )

    if len(daily) < TOP_N:
        continue

    top = (
        daily
        .sort_values(
            "AccumulationScore",
            ascending=False
        )
        .head(TOP_N)
    )

    row = {
        "Date": date
    }

    valid = True

    for hold_days in HOLD_DAYS_LIST:

        col = f"FutureRet_{hold_days}"

        future_ret = top[col]

        if future_ret.isna().all():

            valid = False
            break

        row[f"Ret_{hold_days}"] = (
            future_ret.mean()
        )

    if valid:

        results.append(row)

# =========================
# 集計
# =========================
result_df = pd.DataFrame(results)

# =========================
# 出力
# =========================
print("\n===================")
print("BACKTEST RESULT")
print("===================")

for hold_days in HOLD_DAYS_LIST:

    ret_col = f"Ret_{hold_days}"

    mean_ret = (
        result_df[ret_col]
        .mean()
    )

    median_ret = (
        result_df[ret_col]
        .median()
    )

    win_rate = (
        result_df[ret_col] > 0
    ).mean()

    annualized = (
        (1 + mean_ret)
        ** (252 / hold_days)
        - 1
    )

    print(
        f"\nHold {hold_days} Days"
    )

    print(
        f"Mean Return   : {mean_ret:.2%}"
    )

    print(
        f"Median Return : {median_ret:.2%}"
    )

    print(
        f"Win Rate      : {win_rate:.2%}"
    )

    print(
        f"Annualized    : {annualized:.2%}"
    )

# =========================
# TOP5確認
# =========================
latest_date = (
    df["Date"].max()
)

latest = (
    df[
        df["Date"]
        == latest_date
    ]
    .sort_values(
        "AccumulationScore",
        ascending=False
    )
)

print("\n===================")
print("LATEST TOP 20")
print("===================")

print(
    latest[
        [
            "Ticker",
            "Name",
            "Industry",
            "AccumulationScore",
            "IndustryRankPct"
        ]
    ]
    .head(20)
)

# =========================
# 保存
# =========================
result_df.to_csv(
    "stock_data/backtest_result.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nSaved: stock_data/backtest_result.csv"
)
import pandas as pd
import numpy as np

# =========================
# 設定
# =========================
DATA_PATH = "stock_data/technical_features.parquet"

TEST_WINDOWS = [
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
]

HOLD_DAYS = 5
MAX_POSITIONS = 10

SLIPPAGE = 0.002
COMMISSION = 0.001

STRONG_WEIGHT = 2.0
WEAK_WEIGHT = 1.0

# =========================
# load
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

# =========================
# forward return（評価用のみ）
# =========================
df["forward_return"] = (
    df.groupby("Ticker")["Close"].shift(-HOLD_DAYS) / df["Close"] - 1
)

# =========================
# signal_score（前工程で作られている前提）
# =========================
df = pd.read_parquet("stock_data/technical_features.parquet")

df["signal_score"] = df.apply(generate_signal, axis=1)

# =========================
# cross-sectional ranking（最重要）
# =========================
df["score_rank"] = df.groupby("Date")["signal_score"].rank(pct=True)

# =========================
# 強弱分類（動的）
# =========================
df["signal_type"] = 0

df.loc[df["score_rank"] >= 0.8, "signal_type"] = 2  # strong
df.loc[(df["score_rank"] >= 0.5) & (df["score_rank"] < 0.8), "signal_type"] = 1  # weak

# =========================
# backtest
# =========================
all_results = []

for start, end in TEST_WINDOWS:

    print(f"\n===== WINDOW {start} → {end} =====")

    d = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    dates = sorted(d["Date"].unique())

    positions = []
    trade_log = []

    for date in dates:

        # =========================
        # exit処理
        # =========================
        new_positions = []

        for p in positions:

            hold = (date - p["entry"]).days

            if hold >= HOLD_DAYS:

                ret = p["ret"]
                ret -= (SLIPPAGE + COMMISSION)

                trade_log.append({
                    "Date": date,
                    "Ticker": p["Ticker"],
                    "Return": ret,
                    "signal_type": p["signal_type"],
                    "weight": p["weight"]
                })

            else:
                new_positions.append(p)

        positions = new_positions

        # =========================
        # entry（ランキングのみ）
        # =========================
        today = d[d["Date"] == date]

        # TOP制限なし（重要）
        candidates = today[today["signal_type"] > 0].copy()

        if len(candidates) == 0:
            continue

        # 強い順にソート
        candidates = candidates.sort_values("score_rank", ascending=False)

        # ポジション枠
        slots = MAX_POSITIONS - len(positions)

        if slots <= 0:
            continue

        # =========================
        # エントリー
        # =========================
        for _, row in candidates.head(slots).iterrows():

            if row["signal_type"] == 2:
                weight = STRONG_WEIGHT
            else:
                weight = WEAK_WEIGHT

            positions.append({
                "Ticker": row["Ticker"],
                "entry": date,
                "ret": row["forward_return"],
                "signal_type": row["signal_type"],
                "weight": weight
            })

    # =========================
    # result
    # =========================
    trade_df = pd.DataFrame(trade_log)

    if len(trade_df) == 0:
        print("No trades")
        continue

    # 重み付きリターン
    trade_df["weighted_return"] = trade_df["Return"] * trade_df["weight"]

    print("\n===== RESULT =====")
    print(trade_df["weighted_return"].describe())

    print("\nWin Rate:", (trade_df["weighted_return"] > 0).mean())

    print("\nAverage Return:", trade_df["weighted_return"].mean())

    print("\nSharpe:",
          trade_df["weighted_return"].mean() /
          (trade_df["weighted_return"].std() + 1e-9))

    print("\n===== SIGNAL TYPE BREAKDOWN =====")
    print(trade_df["signal_type"].value_counts())

    all_results.append(trade_df)

# =========================
# global
# =========================
full = pd.concat(all_results)

print("\n===== GLOBAL SUMMARY =====")
print(full["weighted_return"].describe())
print("\nGlobal Win Rate:", (full["weighted_return"] > 0).mean())
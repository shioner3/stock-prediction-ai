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

MAX_POSITIONS = 5
HOLD_DAYS = 5
SLIPPAGE = 0.002
COMMISSION = 0.001

# =========================
# load
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

# =========================
# signal engine（改善版）
# =========================
def generate_signal(row):

    score = 0

    # ===== モメンタム =====
    if row["return_5d"] > 0:
        score += 1
    if row["return_rank_daily"] >= 0.7:
        score += 1

    # ===== トレンド =====
    if row["close_ma5_ratio"] > 1.01:
        score += 1
    if row["close_ma25_ratio"] > 1.00:
        score += 1
    if row["bb_position"] > 0.65:
        score += 1

    # ===== 出来高 =====
    if row["volume_ratio_5d"] > 1.0:
        score += 1
    if row["volume_ratio_20d"] > 0.9:
        score += 0.5
    if row["volume_rank_daily"] > 0.4:
        score += 1

    # ===== ブレイク =====
    if row["high_break_20d"] > 0:
        score += 1
    if row["bb_position"] > 0.7:
        score += 0.5

    # ===== リスク =====
    if row["gap_up_ratio"] < 0.05:
        score += 0.5
    if row["atr_ratio"] < 0.12:
        score += 0.5
    if row["range_compression_5d"] > 0.95:
        score += 0.5

    return score


# =========================
# backtest
# =========================
for start, end in TEST_WINDOWS:

    print(f"\n===== WINDOW {start} → {end} =====")

    d = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    # =========================
    # signal生成（スコア化）
    # =========================
    d["score"] = d.apply(generate_signal, axis=1)

    # 日次ランキング（これが本体）
    d["rank"] = d.groupby("Date")["score"].rank(ascending=False, method="first")

    # =========================
    # backtest
    # =========================
    dates = sorted(d["Date"].unique())

    positions = []
    trade_log = []

    for date in dates:

        new_positions = []

        # ===== exit =====
        for p in positions:

            hold = (date - p["entry"]).days

            if hold >= HOLD_DAYS:

                ret = p["ret"]
                ret -= (SLIPPAGE + COMMISSION)

                trade_log.append({
                    "Date": date,
                    "Ticker": p["Ticker"],
                    "Return": ret,
                    "score": p["score"]
                })

            else:
                new_positions.append(p)

        positions = new_positions

        # ===== entry =====
        slots = MAX_POSITIONS - len(positions)

        if slots > 0:

            picks = d[d["Date"] == date].sort_values(
                "score",
                ascending=False
            ).head(slots)

            for _, row in picks.iterrows():

                future_ret = row.get("return_5d", 0)

                positions.append({
                    "Ticker": row["Ticker"],
                    "entry": date,
                    "ret": future_ret,
                    "score": row["score"]
                })

    # =========================
    # result
    # =========================
    trade_df = pd.DataFrame(trade_log)

    if len(trade_df) == 0:
        print("No trades")
        continue

    print("\n===== RESULT =====")
    print(trade_df["Return"].describe())

    print("\nWin Rate:", (trade_df["Return"] > 0).mean())

    print("\nAverage Return:", trade_df["Return"].mean())

    print("\nSharpe (simple):",
          trade_df["Return"].mean() / (trade_df["Return"].std() + 1e-9))

    print("\n===== SCORE STATS =====")
    print(trade_df["score"].describe())
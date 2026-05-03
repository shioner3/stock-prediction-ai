import pandas as pd
import numpy as np

DATA_PATH = "stock_data/signals.parquet"

HOLD_DAYS = 5
MAX_POSITIONS = 10

SLIPPAGE = 0.002
COMMISSION = 0.001

# ★調整ポイント（トレード増やす方向）
ENTRY_THRESHOLD = 0.15   # ↓下げる（重要）
STRONG_THRESHOLD = 1.5

df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

# =========================
# backtest
# =========================
dates = sorted(df["Date"].unique())

positions = []
logs = []

# ★追加：統計用
trade_count_per_day = []

for date in dates:

    # =====================
    # exit
    # =====================
    new_positions = []

    for p in positions:

        hold = (date - p["entry"]).days

        if hold >= HOLD_DAYS:

            ret = p["ret"] - (SLIPPAGE + COMMISSION)

            logs.append({
                "Date": date,
                "Return": ret,
                "signal": p["signal"],
                "score": p["score"]
            })

        else:
            new_positions.append(p)

    positions = new_positions

    # =====================
    # entry
    # =====================
    today = df[df["Date"] == date].copy()

    # ---------------------
    # ① threshold（緩めてトレード増やす）
    # ---------------------
    candidates = today[
        today["signal_score"] > ENTRY_THRESHOLD
    ].copy()

    # ★追加：候補数記録
    trade_count_per_day.append(len(candidates))

    if len(candidates) == 0:
        continue

    # ---------------------
    # ② ranking
    # ---------------------
    candidates = candidates.sort_values(
        ["signal_score", "ret_rank"],
        ascending=False
    )

    slots = MAX_POSITIONS - len(positions)

    if slots <= 0:
        continue

    # ---------------------
    # ③ entry
    # ---------------------
    for _, row in candidates.head(slots).iterrows():

        positions.append({
            "entry": date,
            "ret": row["forward_return"],
            "signal": row["signal_entry"],
            "score": row["signal_score"]
        })

# =========================
# result
# =========================
res = pd.DataFrame(logs)

print("\n===== RESULT =====")
print(res["Return"].describe())

print("\nWin Rate:", (res["Return"] > 0).mean())

print("\nSharpe:",
      res["Return"].mean() / (res["Return"].std() + 1e-9))

# =========================
# ★トレード数分析（重要追加）
# =========================
print("\n===== TRADE STATISTICS =====")

print("Total Trades:", len(res))
print("Avg Trades Per Day:", np.mean(trade_count_per_day))
print("Max Trades Per Day:", np.max(trade_count_per_day))
print("Min Trades Per Day:", np.min(trade_count_per_day))

print("\nActive Days:", len([x for x in trade_count_per_day if x > 0]))
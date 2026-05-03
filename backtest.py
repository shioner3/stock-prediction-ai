import pandas as pd
import numpy as np

DATA_PATH = "stock_data/signals.parquet"

HOLD_DAYS = 5
MAX_POSITIONS = 10

SLIPPAGE = 0.002
COMMISSION = 0.001

df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(["Date", "Ticker"])

# =========================
# backtest
# =========================
dates = sorted(df["Date"].unique())

positions = []
logs = []

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
                "signal": p["signal"]
            })

        else:
            new_positions.append(p)

    positions = new_positions

    # =====================
    # entry（TOP固定禁止）
    # =====================
    today = df[df["Date"] == date]

    candidates = today[today["signal_entry"] > 0].copy()

    if len(candidates) == 0:
        continue

    # ランダム性防止でscore順
    candidates = candidates.sort_values("signal_score", ascending=False)

    slots = MAX_POSITIONS - len(positions)

    for _, row in candidates.head(slots).iterrows():

        positions.append({
            "entry": date,
            "ret": row["forward_return"],
            "signal": row["signal_entry"]
        })

# =========================
# result
# =========================
res = pd.DataFrame(logs)

print("\n===== RESULT =====")
print(res["Return"].describe())

print("\nWin Rate:", (res["Return"] > 0).mean())

print("\nSharpe:", res["Return"].mean() / (res["Return"].std() + 1e-9))
import pandas as pd
import numpy as np

DATA_PATH = "stock_data/signals.parquet"

HOLD_DAYS = 5
MAX_POSITIONS = 5

MAX_NEW_ENTRIES_PER_DAY = 2   # ★重要
ENTRY_THRESHOLD = 0.25        # ★厳しめ

SLIPPAGE = 0.002
COMMISSION = 0.001

df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

dates = sorted(df["Date"].unique())

positions = []
logs = []

for date in dates:

    # =====================
    # exit（固定 + 劣化判定）
    # =====================
    new_positions = []

    for p in positions:

        hold = (date - p["entry"]).days

        # 固定ホールド or スコア劣化でexit
        if hold >= HOLD_DAYS or p["score"] < ENTRY_THRESHOLD:

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

    # ★① 厳選候補
    candidates = today[today["signal_score"] > ENTRY_THRESHOLD]

    if len(candidates) == 0:
        continue

    # ★② 上位だけ
    candidates = candidates.sort_values("signal_score", ascending=False).head(50)

    # ★③ さらに制限
    candidates = candidates[candidates["ret_rank"] > 0.8]

    slots = MAX_POSITIONS - len(positions)
    if slots <= 0:
        continue

    # ★④ 1日最大エントリー制限
    entry_count = 0

    for _, row in candidates.iterrows():

        if entry_count >= MAX_NEW_ENTRIES_PER_DAY:
            break

        if len(positions) >= MAX_POSITIONS:
            break

        positions.append({
            "entry": date,
            "ret": row["forward_return"],
            "signal": row["signal_entry"],
            "score": row["signal_score"]
        })

        entry_count += 1


# =========================
# result
# =========================
res = pd.DataFrame(logs)

print("\n===== RESULT =====")
print(res["Return"].describe())

print("\nWin Rate:", (res["Return"] > 0).mean())

print("\nSharpe:",
      res["Return"].mean() / (res["Return"].std() + 1e-9))

print("\n===== STATS =====")
print("Total Trades:", len(res))
print("Avg Trade Return:", res["Return"].mean())
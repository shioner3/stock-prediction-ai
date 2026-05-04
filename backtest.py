import pandas as pd
import numpy as np

# =========================
# 設定
# =========================
DATA_PATH = "stock_data/signals.parquet"

HOLD_DAYS = 5
MAX_POSITIONS = 5

MAX_NEW_ENTRIES_PER_DAY = 2

SLIPPAGE = 0.002
COMMISSION = 0.001

REGIME_SPLITS = [
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
]

TOP_Q = 0.2
MAX_TURNOVER_PER_DAY = 0.3

# =========================
# load
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

all_logs = []

# =========================
# walk-forward
# =========================
for start, end in REGIME_SPLITS:

    print(f"\n===== REGIME {start} → {end} =====")

    d = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    dates = sorted(d["Date"].unique())

    positions = []
    logs = []

    for date in dates:

        today = d[d["Date"] == date].copy()

        # =========================
        # ★ DEBUG① entry前状態
        # =========================
        if len(today) > 0:
            print(f"\n[DEBUG] {date}")
            print("rows:", len(today))

            if "forward_return" in today.columns:
                print("forward_return mean:", today["forward_return"].mean())

            if "signal_score" in today.columns:
                print("signal_score mean:", today["signal_score"].mean())
                print("signal_score max:", today["signal_score"].max())

        # 空日はスキップ
        if len(today) == 0:
            continue

        # =========================
        # exit
        # =========================
        new_positions = []
        turnover_cost = 0.0

        for p in positions:

            hold = (date - p["entry"]).days

            if hold >= HOLD_DAYS:

                ret = p["ret"] - (SLIPPAGE + COMMISSION)

                logs.append({
                    "Date": date,
                    "Return": ret,
                    "signal": p["signal"],
                    "edge": p["edge"],
                    "regime": start
                })

                turnover_cost += abs(ret) * 0.01

            else:
                new_positions.append(p)

        positions = new_positions

        # =========================
        # entry前デバッグ②
        # =========================
        print("pre-entry len(today):", len(today))

        # =========================
        # ranking（ここで一回だけ作る）
        # =========================
        today["score_rank"] = today["signal_score"].rank(pct=True)
        today["return_rank"] = today["forward_return"].rank(pct=True)

        today["edge_score"] = (
            0.7 * today["score_rank"] +
            0.3 * today["return_rank"]
        )

        # =========================
        # entry filter（動的）
        # =========================
        threshold = today["score_rank"].quantile(1 - TOP_Q)
        candidates = today[today["score_rank"] >= threshold].copy()

        if len(candidates) == 0:
            continue

        candidates = candidates.sort_values("edge_score", ascending=False)

        slots = MAX_POSITIONS - len(positions)
        if slots <= 0:
            continue

        entry_count = 0
        used = set()

        for _, row in candidates.iterrows():

            if entry_count >= MAX_NEW_ENTRIES_PER_DAY:
                break

            if len(positions) >= MAX_POSITIONS:
                break

            if row["Ticker"] in used:
                continue

            if turnover_cost > MAX_TURNOVER_PER_DAY:
                break

            positions.append({
                "entry": date,
                "ret": row["forward_return"],
                "signal": row["signal_entry"],
                "edge": row["edge_score"],
                "Ticker": row["Ticker"]
            })

            used.add(row["Ticker"])
            entry_count += 1

        logs.append({
            "Date": date,
            "Return": np.nan,
            "signal": -1,
            "edge": -1,
            "turnover": len(positions) / MAX_POSITIONS,
            "regime": start
        })

    all_logs.extend(logs)

# =========================
# result
# =========================
res = pd.DataFrame(all_logs).dropna()

print("\n===== RESULT =====")
print(res["Return"].describe())

print("\nWin Rate:", (res["Return"] > 0).mean())

print("\nSharpe:",
      res["Return"].mean() / (res["Return"].std() + 1e-9))

print("\n===== RISK =====")
print("Avg Turnover:", res["turnover"].mean())
print("Max Turnover:", res["turnover"].max())

print("\n===== STATS =====")
print("Total Trades:", len(res))
print("Avg Trade Return:", res["Return"].mean())
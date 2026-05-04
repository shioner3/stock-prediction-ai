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
trade_logs = []

# =========================
# payoff推定
# =========================
pos_mean = df[df["forward_return"] > 0]["forward_return"].mean()
neg_mean = abs(df[df["forward_return"] < 0]["forward_return"].mean())

if np.isnan(pos_mean):
    pos_mean = 0.05
if np.isnan(neg_mean):
    neg_mean = 0.03

print("\n===== PAYOFF STRUCTURE =====")
print("avg gain:", pos_mean)
print("avg loss:", neg_mean)

# =========================
# walk-forward
# =========================
for start, end in REGIME_SPLITS:

    print(f"\n===== REGIME {start} → {end} =====")

    d = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    dates = sorted(d["Date"].unique())

    positions = []
    regime_logs = []

    for date in dates:

        today = d[d["Date"] == date].copy()
        if len(today) == 0:
            continue

        # =========================
        # exit
        # =========================
        new_positions = []
        realized_returns = []
        turnover_cost = 0.0

        for p in positions:

            hold = (date - p["entry"]).days

            if hold >= HOLD_DAYS:

                ret = p["ret"] - (SLIPPAGE + COMMISSION)
                realized_returns.append(ret)

                trade_logs.append({
                    "entry_date": p["entry"],
                    "exit_date": date,
                    "Return": ret,
                    "prob": p["prob"],
                    "ev": p["ev"],
                    "weight": p["weight"]
                })

                turnover_cost += abs(ret) * 0.01

            else:
                new_positions.append(p)

        positions = new_positions

        # =========================
        # ★ prob（rankベース）
        # =========================
        rank = today["signal_score"].rank(pct=True)

        # ★重要：非線形で差を広げる
        today["prob_entry"] = rank ** 2

        # =========================
        # expected value
        # =========================
        today["expected_return"] = (
            today["prob_entry"] * pos_mean -
            (1 - today["prob_entry"]) * neg_mean
        )

        # =========================
        # position sizing
        # =========================
        today["position_size"] = np.clip(today["expected_return"], 0, None)

        if today["position_size"].sum() > 0:
            today["position_size"] /= today["position_size"].sum()

        # =========================
        # entry universe
        # =========================
        threshold = today["prob_entry"].quantile(1 - TOP_Q)
        candidates = today[today["prob_entry"] >= threshold].copy()

        if len(candidates) == 0:
            continue

        candidates = candidates.sort_values("expected_return", ascending=False)

        # =========================
        # entry
        # =========================
        entry_count = 0

        for _, row in candidates.iterrows():

            if entry_count >= MAX_NEW_ENTRIES_PER_DAY:
                break

            if len(positions) >= MAX_POSITIONS:
                break

            if turnover_cost > MAX_TURNOVER_PER_DAY:
                break

            positions.append({
                "entry": date,
                "ret": row["forward_return"],
                "Ticker": row["Ticker"],
                "weight": row["position_size"],
                "prob": row["prob_entry"],
                "ev": row["expected_return"]
            })

            entry_count += 1

        # =========================
        # daily summary
        # =========================
        regime_logs.append({
            "Date": date,
            "n_positions": len(positions),
            "n_entries": entry_count,
            "mean_signal": today["signal_score"].mean(),
            "mean_prob": today["prob_entry"].mean(),
            "mean_ev": today["expected_return"].mean(),
            "realized_return_sum": np.sum(realized_returns) if len(realized_returns) > 0 else 0.0
        })

    all_logs.extend(regime_logs)

# =========================
# result
# =========================
res = pd.DataFrame(all_logs)

print("\n===== RESULT =====")
print(res["realized_return_sum"].describe())

print("\n===== STATS =====")
print("Avg positions:", res["n_positions"].mean())
print("Avg entries/day:", res["n_entries"].mean())
print("Avg probability:", res["mean_prob"].mean())
print("Avg expected value:", res["mean_ev"].mean())

# =========================
# 勝ち vs 負け分析
# =========================
tr = pd.DataFrame(trade_logs)

win = tr[tr["Return"] > 0]
lose = tr[tr["Return"] <= 0]

print("\n===== WIN vs LOSE =====")

print("\n--- COUNT ---")
print("win:", len(win), "lose:", len(lose))

print("\n--- MEAN ---")
print("WIN")
print(win[["Return", "prob", "ev", "weight"]].mean())

print("\nLOSE")
print(lose[["Return", "prob", "ev", "weight"]].mean())

print("\n--- DISTRIBUTION ---")
for c in ["prob", "ev", "weight"]:
    print(f"\n=== {c} ===")
    print("WIN:", win[c].describe())
    print("LOSE:", lose[c].describe())

# =========================
# DEBUG
# =========================
print("\n===== DEBUG RETURN =====")
print(df["forward_return"].describe())
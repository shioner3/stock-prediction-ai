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
# sigmoid
# =========================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
        # exit（評価だけforward_return使用）
        # =========================
        new_positions = []
        realized_returns = []
        turnover_cost = 0.0

        for p in positions:

            hold = (date - p["entry"]).days

            if hold >= HOLD_DAYS:

                ret = p["ret"] - (SLIPPAGE + COMMISSION)
                realized_returns.append(ret)

                turnover_cost += abs(ret) * 0.01

            else:
                new_positions.append(p)

        positions = new_positions

        # =========================
        # ① signal → probability
        # =========================
        score = today["signal_score"]
        score_z = (score - score.mean()) / (score.std() + 1e-9)

        today["prob_entry"] = sigmoid(score_z)

        # =========================
        # ② expected_return（未来使わない）
        # =========================
        # ★ proxy設計（重要）
        today["expected_return"] = (
            0.5 * today["prob_entry"] +
            0.3 * score_z +
            0.2 * today["signal_score"]
        )

        # =========================
        # ③ position sizing（未来なし）
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
                "ret": row["forward_return"],  # ←ここだけOK（exit専用）
                "Ticker": row["Ticker"],
                "weight": row["position_size"],
                "prob": row["prob_entry"]
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
            "mean_expected_return": today["expected_return"].mean(),
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
print("Avg signal:", res["mean_signal"].mean())
print("Avg probability:", res["mean_prob"].mean())
print("Avg expected return:", res["mean_expected_return"].mean())
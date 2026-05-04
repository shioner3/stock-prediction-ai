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

    # ===== 集計用ログ（圧縮） =====
    regime_logs = []

    for date in dates:

        today = d[d["Date"] == date].copy()

        # =========================
        # ★圧縮デバッグ（異常時のみ）
        # =========================
        if len(today) > 0:
            fr = today["forward_return"]
            ss = today["signal_score"]

            # 異常検知だけ出す
            if fr.isna().mean() > 0.5 or ss.std() < 1e-6:
                print(f"[WARN] {date}")
                print("forward_return nan%:", fr.isna().mean())
                print("signal_score std:", ss.std())

        if len(today) == 0:
            continue

        # =========================
        # exit
        # =========================
        new_positions = []
        turnover_cost = 0.0

        realized_returns = []

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
        # entry準備
        # =========================
        today["score_rank"] = today["signal_score"].rank(pct=True)
        today["return_rank"] = today["forward_return"].rank(pct=True)

        today["edge_score"] = (
            0.7 * today["score_rank"] +
            0.3 * today["return_rank"]
        )

        threshold = today["score_rank"].quantile(1 - TOP_Q)
        candidates = today[today["score_rank"] >= threshold]

        if len(candidates) == 0:
            continue

        candidates = candidates.sort_values("edge_score", ascending=False)

        slots = MAX_POSITIONS - len(positions)
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
                "Ticker": row["Ticker"]
            })

            entry_count += 1

        # =========================
        # 日次ログ（圧縮）
        # =========================
        regime_logs.append({
            "Date": date,
            "n_positions": len(positions),
            "n_entries": entry_count,
            "mean_score": today["signal_score"].mean(),
            "mean_forward_return": today["forward_return"].mean(),
            "realized_return_sum": np.sum(realized_returns) if len(realized_returns) > 0 else 0.0
        })

    all_logs.extend(regime_logs)

# =========================
# result
# =========================
res = pd.DataFrame(all_logs)

print("\n===== RESULT =====")
print(res[["realized_return_sum"]].describe())

print("\n===== STATS =====")
print("Avg positions:", res["n_positions"].mean())
print("Avg entries/day:", res["n_entries"].mean())
print("Avg signal:", res["mean_score"].mean())
print("Avg forward return:", res["mean_forward_return"].mean())
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

# ===== ① walk-forward（レジーム分割）
REGIME_SPLITS = [
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
]

# ===== ② 動的閾値（固定TOP廃止）
TOP_Q = 0.2

# ===== ④ turnover制御
MAX_TURNOVER_PER_DAY = 0.3  # 資産の30%まで入れ替え

# =========================
# load
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

all_logs = []

# =========================
# walk-forward loop
# =========================
for start, end in REGIME_SPLITS:

    print(f"\n===== REGIME {start} → {end} =====")

    d = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    dates = sorted(d["Date"].unique())

    positions = []
    logs = []

    # turnover tracking
    prev_capital = 1.0

    for date in dates:

        today = d[d["Date"] == date].copy()
        if len(today) == 0:
            continue

        # =====================
        # exit
        # =====================
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

                turnover_cost += abs(ret) * 0.01  # 擬似売買コスト

            else:
                new_positions.append(p)

        positions = new_positions
        
        # =====================
        # entryの直前に追加
        # =====================
        today = d[d["Date"] == date].copy()
        
        # ★追加：edge_scoreをその場で生成
        today["score_rank"] = today["signal_score"].rank(pct=True)
        today["return_rank"] = today["forward_return"].rank(pct=True)

        today["edge_score"] = (
            0.7 * today["score_rank"] +
            0.3 * today["return_rank"]
        )

        # ★追加：クロスセクションrank生成
        today["score_rank"] = today["signal_score"].rank(pct=True)
        
        # =====================
        # entry（動的閾値）
        # =====================

        # ★② TOP固定禁止 → 分位動的
        threshold = today["score_rank"].quantile(1 - TOP_Q)

        candidates = today[today["score_rank"] >= threshold].copy()

        if len(candidates) == 0:
            continue

        # edge順
        candidates = candidates.sort_values("edge_score", ascending=False)

        # =====================
        # ③ 約定モデル（簡易）
        # =====================
        # スリッページ増加（流動性低いほど悪化）
        candidates["exec_slippage"] = SLIPPAGE * (1 + (1 - candidates["score_rank"]))

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

            # =====================
            # ④ turnover制御
            # =====================
            if turnover_cost > MAX_TURNOVER_PER_DAY:
                break

            positions.append({
                "entry": date,
                "ret": row["forward_return"],
                "signal": row["signal_entry"],
                "edge": row["edge_score"],
                "Ticker": row["Ticker"],
                "slip": row["exec_slippage"]
            })

            used.add(row["Ticker"])
            entry_count += 1

        # =====================
        # portfolio risk tracking
        # =====================
        pos_value = len(positions)
        turnover_ratio = pos_value / MAX_POSITIONS

        logs.append({
            "Date": date,
            "Return": np.nan,
            "signal": -1,
            "edge": -1,
            "turnover": turnover_ratio,
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
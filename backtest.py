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

USE_MARKET_FILTER = True

# =========================
# load
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

# =========================
# ★デバッグ①：market_return存在確認
# =========================
if "market_return" not in df.columns:
    raise ValueError("❌ market_return が存在しない（特徴量で作れてない）")

print("\n===== MARKET RETURN STATS =====")
print(df["market_return"].describe())

# =========================
# 市場トレンド
# =========================
df["market_trend"] = (df["market_return"] > 0).astype(int)

# =========================
# ★デバッグ②：トレンド分布
# =========================
print("\n===== MARKET TREND DISTRIBUTION =====")
print(df["market_trend"].value_counts(normalize=True))

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

    blocked_days = 0
    total_days = 0

    for date in dates:

        today = d[d["Date"] == date].copy()
        if len(today) == 0:
            continue

        total_days += 1

        # =========================
        # 市場フィルタ
        # =========================
        if USE_MARKET_FILTER:
            if today["market_trend"].iloc[0] == 0:
                allow_entry = False
                blocked_days += 1
            else:
                allow_entry = True
        else:
            allow_entry = True

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
                    "weight": p["weight"],
                    "market_return": p["market_return"]
                })

                turnover_cost += abs(ret) * 0.01

            else:
                new_positions.append(p)

        positions = new_positions

        # =========================
        # prob
        # =========================
        today["prob_entry"] = today["signal_score"].rank(pct=True)

        # =========================
        # expected value
        # =========================
        today["expected_return"] = today["prob_entry"] * pos_mean

        # =========================
        # position sizing
        # =========================
        today["position_size"] = np.clip(today["expected_return"], 0, None)

        if today["position_size"].sum() > 0:
            today["position_size"] /= today["position_size"].sum()

        # =========================
        # entry universe
        # =========================
        threshold = today["expected_return"].quantile(1 - TOP_Q)
        candidates = today[today["expected_return"] >= threshold].copy()

        if len(candidates) == 0:
            continue

        candidates = candidates.sort_values("expected_return", ascending=False)

        # =========================
        # entry
        # =========================
        entry_count = 0

        if allow_entry:

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
                    "ev": row["expected_return"],
                    "market_return": row["market_return"]
                })

                entry_count += 1

        # =========================
        # ★デバッグ③：entry抑制確認
        # =========================
        if not allow_entry and entry_count > 0:
            print(f"⚠️ FILTER ERROR {date}: 弱い日にエントリーしてる")

        regime_logs.append({
            "Date": date,
            "n_positions": len(positions),
            "n_entries": entry_count,
            "mean_prob": today["prob_entry"].mean(),
            "mean_ev": today["expected_return"].mean(),
            "realized_return_sum": np.sum(realized_returns) if len(realized_returns) > 0 else 0.0,
            "market_return": today["market_return"].mean(),
            "allow_entry": allow_entry
        })

    print(f"\n--- FILTER STATS ---")
    print("blocked_days:", blocked_days)
    print("total_days:", total_days)
    print("blocked_ratio:", blocked_days / total_days if total_days > 0 else 0)

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
# 勝ち vs 負け
# =========================
tr = pd.DataFrame(trade_logs)

win = tr[tr["Return"] > 0]
lose = tr[tr["Return"] <= 0]

print("\n===== WIN vs LOSE =====")
print("win:", len(win), "lose:", len(lose))

# =========================
# 市場別分析
# =========================
bull = tr[tr["market_return"] > 0]
bear = tr[tr["market_return"] <= 0]

print("\n===== BULL =====")
print(bull["Return"].describe())

print("\n===== BEAR =====")
print(bear["Return"].describe())

# =========================
# ★デバッグ④：市場別勝率
# =========================
print("\n===== MARKET WIN RATE =====")
print("BULL:", (bull["Return"] > 0).mean())
print("BEAR:", (bear["Return"] > 0).mean())

# =========================
# decile分析
# =========================
print("\n===== DECILE ANALYSIS =====")

tr["decile"] = pd.qcut(tr["prob"], 10, labels=False, duplicates="drop")

decile_stats = tr.groupby("decile").agg({
    "Return": ["mean", "count"],
    "prob": "mean",
    "ev": "mean"
})

decile_stats["win_rate"] = tr.groupby("decile")["Return"].apply(lambda x: (x > 0).mean())

print(decile_stats.sort_index())

# =========================
# DEBUG
# =========================
print("\n===== DEBUG RETURN =====")
print(df["forward_return"].describe())
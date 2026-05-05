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

print("\n===== DEBUG: COLUMNS =====")
print(df.columns)

print("\n===== DEBUG: HEAD =====")
print(df.head())

# market_return確認
if "market_return" not in df.columns:
    raise ValueError("❌ market_return が存在しない")

print("\n===== DEBUG: market_return SAMPLE =====")
print(df[["Date", "Ticker", "market_return"]].head())

df = df.sort_values(["Date", "Ticker"])

# =========================
# 市場トレンド
# =========================
df["market_trend"] = (df["market_return"] > 0).astype(int)

print("\n===== MARKET RETURN STATS =====")
print(df["market_return"].describe())

print("\n===== MARKET TREND DISTRIBUTION =====")
print(df["market_trend"].value_counts(normalize=True))

all_logs = []
trade_logs = []

# =========================
# payoff
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

    print("rows:", len(d))

    if len(d) == 0:
        print("⚠️ データなし")
        continue

    dates = sorted(d["Date"].unique())

    positions = []
    regime_logs = []

    blocked_days = 0

    for date in dates:

        today = d[d["Date"] == date].copy()
        if len(today) == 0:
            continue

        # =========================
        # 市場フィルタ
        # =========================
        if USE_MARKET_FILTER:
            allow_entry = today["market_trend"].iloc[0] == 1
        else:
            allow_entry = True

        if not allow_entry:
            blocked_days += 1

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
                    "Return": ret,
                    "prob": p["prob"],
                    "ev": p["ev"],
                    "market_return": p["market_return"]
                })

            else:
                new_positions.append(p)

        positions = new_positions

        # =========================
        # prob
        # =========================
        today["prob_entry"] = today["signal_score"].rank(pct=True)

        # =========================
        # EV
        # =========================
        today["expected_return"] = today["prob_entry"] * pos_mean

        # =========================
        # sizing
        # =========================
        today["position_size"] = np.clip(today["expected_return"], 0, None)

        if today["position_size"].sum() > 0:
            today["position_size"] /= today["position_size"].sum()

        # =========================
        # candidates
        # =========================
        threshold = today["expected_return"].quantile(1 - TOP_Q)
        candidates = today[today["expected_return"] >= threshold]

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

                positions.append({
                    "entry": date,
                    "ret": row["forward_return"],
                    "prob": row["prob_entry"],
                    "ev": row["expected_return"],
                    "market_return": row["market_return"]
                })

                entry_count += 1

        # =========================
        # summary
        # =========================
        regime_logs.append({
            "Date": date,
            "realized_return_sum": np.sum(realized_returns) if realized_returns else 0.0
        })

    print("\n--- FILTER STATS ---")
    print("blocked_days:", blocked_days)
    print("total_days:", len(dates))
    print("blocked_ratio:", blocked_days / len(dates))

    all_logs.extend(regime_logs)

# =========================
# result
# =========================
res = pd.DataFrame(all_logs)

print("\n===== RESULT =====")

if len(res) == 0:
    print("❌ resが空 → データが流れてない")
else:
    print(res["realized_return_sum"].describe())

# =========================
# trade analysis
# =========================
tr = pd.DataFrame(trade_logs)

if len(tr) == 0:
    print("\n❌ トレード0件")
else:
    print("\n===== BULL / BEAR =====")
    print(tr.groupby(tr["market_return"] > 0)["Return"].describe())
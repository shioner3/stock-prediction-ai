import pandas as pd
import numpy as np

# =========================
# 設定（実運用寄り）
# =========================
DATA_PATH = "stock_data/signals.parquet"

HOLD_DAYS = 5
MAX_POSITIONS = 5

MAX_NEW_ENTRIES_PER_DAY = 2

TOP_Q = 0.2            # 上位20%のみ対象
DIVERSITY_LIMIT = 2    # 同一セクター代替用（なければTickerで代用）

SLIPPAGE = 0.002
COMMISSION = 0.001

# =========================
# load
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

# =========================
# ★重要：日次ランキングに統一（絶対値禁止）
# =========================
df["score_rank"] = df.groupby("Date")["signal_score"].rank(pct=True)

df["return_rank"] = df.groupby("Date")["forward_return"].rank(pct=True)

# =========================
# エントリー候補（期待値ベース）
# =========================
df["edge_score"] = (
    0.7 * df["score_rank"] +
    0.3 * df["return_rank"]
)

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
                "signal": p["signal"],
                "edge": p["edge"]
            })

        else:
            new_positions.append(p)

    positions = new_positions

    # =====================
    # entry
    # =====================
    today = df[df["Date"] == date].copy()

    if len(today) == 0:
        continue

    # ★① 上位だけ（絶対閾値禁止）
    candidates = today[today["score_rank"] >= (1 - TOP_Q)]

    if len(candidates) == 0:
        continue

    # ★② エッジ順
    candidates = candidates.sort_values("edge_score", ascending=False)

    slots = MAX_POSITIONS - len(positions)
    if slots <= 0:
        continue

    entry_count = 0
    used_tickers = set()

    # =====================
    # entry制御（重要）
    # =====================
    for _, row in candidates.iterrows():

        if entry_count >= MAX_NEW_ENTRIES_PER_DAY:
            break

        if len(positions) >= MAX_POSITIONS:
            break

        # 重複銘柄制限（実運用必須）
        if row["Ticker"] in used_tickers:
            continue

        positions.append({
            "entry": date,
            "ret": row["forward_return"],
            "signal": row["signal_entry"],
            "edge": row["edge_score"],
            "Ticker": row["Ticker"]
        })

        used_tickers.add(row["Ticker"])
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
print("Avg Trades / Day:", len(res) / len(dates))
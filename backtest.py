import pandas as pd
import numpy as np

# =========================
# 設定
# =========================
DATA_PATH = "stock_data/technical_features.parquet"

TEST_WINDOWS = [
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
]

MAX_POSITIONS = 5
HOLD_DAYS = 5
SLIPPAGE = 0.002
COMMISSION = 0.001

# =========================
# load
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

# =========================
# signal engine（改善版）
# =========================
def generate_signal(row):

    score = 0

    # ===== モメンタム =====
    if row["return_5d"] > 0:
        score += 1
    if row["return_rank_daily"] >= 0.7:
        score += 1

    # ===== トレンド =====
    if row["close_ma5_ratio"] > 1.01:
        score += 1
    if row["close_ma25_ratio"] > 1.00:
        score += 1
    if row["bb_position"] > 0.65:
        score += 1

    # ===== 出来高 =====
    if row["volume_ratio_5d"] > 1.0:
        score += 1
    if row["volume_rank_daily"] > 0.4:
        score += 1

    # ===== ブレイク =====
    if row["high_break_20d"] > 0:
        score += 1

    # ===== リスク =====
    if row["gap_up_ratio"] < 0.05:
        score += 0.5
    if row["atr_ratio"] < 0.12:
        score += 0.5

    return score


# =========================
# 分解用シグナル定義
# =========================
def signal_breakdown(row):
    return {
        "momentum": int(row["return_5d"] > 0 and row["return_rank_daily"] >= 0.7),
        "trend": int(
            row["close_ma5_ratio"] > 1.01 and
            row["close_ma25_ratio"] > 1.00 and
            row["bb_position"] > 0.65
        ),
        "volume": int(
            row["volume_ratio_5d"] > 1.0 and
            row["volume_rank_daily"] > 0.4
        ),
        "breakout": int(row["high_break_20d"] > 0),
        "risk_ok": int(
            row["gap_up_ratio"] < 0.05 and
            row["atr_ratio"] < 0.12
        )
    }


# =========================
# backtest
# =========================
all_signal_stats = []

for start, end in TEST_WINDOWS:

    print(f"\n===== WINDOW {start} → {end} =====")

    d = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    # score
    d["score"] = d.apply(generate_signal, axis=1)

    d["rank"] = d.groupby("Date")["score"].rank(ascending=False, method="first")

    # シグナル分解列を追加
    breakdown = d.apply(signal_breakdown, axis=1, result_type="expand")
    d = pd.concat([d, breakdown], axis=1)

    # =========================
    # backtest
    # =========================
    dates = sorted(d["Date"].unique())

    positions = []
    trade_log = []

    for date in dates:

        new_positions = []

        for p in positions:

            hold = (date - p["entry"]).days

            if hold >= HOLD_DAYS:

                ret = p["ret"]
                ret -= (SLIPPAGE + COMMISSION)

                trade_log.append({
                    "Date": date,
                    "Ticker": p["Ticker"],
                    "Return": ret,
                    "score": p["score"],
                    "momentum": p["momentum"],
                    "trend": p["trend"],
                    "volume": p["volume"],
                    "breakout": p["breakout"],
                    "risk_ok": p["risk_ok"],
                })

            else:
                new_positions.append(p)

        positions = new_positions

        slots = MAX_POSITIONS - len(positions)

        if slots > 0:

            picks = d[d["Date"] == date].sort_values(
                "score", ascending=False
            ).head(slots)

            for _, row in picks.iterrows():

                positions.append({
                    "Ticker": row["Ticker"],
                    "entry": date,
                    "ret": row.get("return_5d", 0),
                    "score": row["score"],
                    "momentum": row["momentum"],
                    "trend": row["trend"],
                    "volume": row["volume"],
                    "breakout": row["breakout"],
                    "risk_ok": row["risk_ok"],
                })

    # =========================
    # 分析
    # =========================
    trade_df = pd.DataFrame(trade_log)

    if len(trade_df) == 0:
        print("No trades")
        continue

    print("\n===== RESULT =====")
    print(trade_df["Return"].describe())

    print("\nWin Rate:", (trade_df["Return"] > 0).mean())
    print("Average Return:", trade_df["Return"].mean())

    print("\nSharpe:",
          trade_df["Return"].mean() / (trade_df["Return"].std() + 1e-9))

    # =========================
    # 🔥 シグナル別分解（ここが重要）
    # =========================
    print("\n===== SIGNAL BREAKDOWN =====")

    for col in ["momentum", "trend", "volume", "breakout", "risk_ok"]:

        win = trade_df[trade_df["Return"] > 0][col].mean()
        lose = trade_df[trade_df["Return"] <= 0][col].mean()

        print(f"{col}: WIN={win:.3f}, LOSE={lose:.3f}, DIFF={win-lose:.3f}")

    all_signal_stats.append(trade_df)

# =========================
# 全体統合
# =========================
full = pd.concat(all_signal_stats)

print("\n===== GLOBAL SUMMARY =====")
print(full["Return"].describe())
print("\nGlobal Win Rate:", (full["Return"] > 0).mean())
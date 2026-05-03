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

# 強度配分
STRONG_WEIGHT = 2.0
WEAK_WEIGHT = 1.0

# =========================
# load
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

# =========================
# signal engine（2段階版）
# =========================
def generate_signal(row):

    momentum = (
        row["return_5d"] > 0 and
        row["return_rank_daily"] >= 0.7
    )

    trend = (
        row["close_ma5_ratio"] > 1.01 and
        row["close_ma25_ratio"] > 1.00 and
        row["bb_position"] > 0.65
    )

    volume = (
        row["volume_ratio_5d"] > 1.0 and
        row["volume_ratio_20d"] > 0.9 and
        row["volume_zscore"] > -0.5 and
        row["volume_rank_daily"] > 0.4
    )

    breakout = (
        (row["high_break_20d"] > 0) or
        (row["bb_position"] > 0.7)
    )

    risk_ok = (
        row["gap_up_ratio"] < 0.05 and
        row["atr_ratio"] < 0.12 and
        row["range_compression_5d"] > 0.95
    )

    # =========================
    # 強シグナル（本命）
    # =========================
    strong = (
        momentum and trend and volume and risk_ok
    )

    # =========================
    # 弱シグナル（準エントリー）
    # =========================
    weak = (
        momentum and trend and breakout
    )

    if strong:
        return 2  # strong

    elif weak:
        return 1  # weak

    else:
        return 0  # no signal


# =========================
# backtest
# =========================
for start, end in TEST_WINDOWS:

    print(f"\n===== WINDOW {start} → {end} =====")

    d = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    # =========================
    # signal生成
    # =========================
    d["signal"] = d.apply(generate_signal, axis=1)

    # 強い順でランキング
    d["score"] = d["signal"]

    d["rank"] = d.groupby("Date")["score"].rank(ascending=False, method="first")

    # =========================
    # backtest
    # =========================
    dates = sorted(d["Date"].unique())

    positions = []
    trade_log = []

    for date in dates:

        new_positions = []

        # =====================
        # exit処理
        # =====================
        for p in positions:

            hold = (date - p["entry"]).days

            if hold >= HOLD_DAYS:

                ret = p["ret"]
                ret -= (SLIPPAGE + COMMISSION)

                trade_log.append({
                    "Date": date,
                    "Ticker": p["Ticker"],
                    "Return": ret,
                    "signal": p["signal"]
                })

            else:
                new_positions.append(p)

        positions = new_positions

        # =====================
        # entry処理
        # =====================
        slots = MAX_POSITIONS - len(positions)

        if slots > 0:

            picks = d[d["Date"] == date]

            # 強 → 弱の順
            picks = picks.sort_values(
                ["signal", "rank"],
                ascending=[False, True]
            ).head(slots)

            for _, row in picks.iterrows():

                # 仮リターン（後で価格ベースに差し替え可）
                future_ret = row.get("return_5d", 0)

                positions.append({
                    "Ticker": row["Ticker"],
                    "entry": date,
                    "ret": future_ret,
                    "signal": row["signal"]
                })

    # =========================
    # 結果
    # =========================
    trade_df = pd.DataFrame(trade_log)

    if len(trade_df) == 0:
        print("No trades")
        continue

    print("\n===== RESULT =====")
    print(trade_df["Return"].describe())

    print("\nWin Rate:", (trade_df["Return"] > 0).mean())

    print("\nAverage Return:", trade_df["Return"].mean())

    print("\nSharpe (simple):",
          trade_df["Return"].mean() / (trade_df["Return"].std() + 1e-9))

    print("\n===== SIGNAL BREAKDOWN =====")
    print(trade_df["signal"].value_counts())
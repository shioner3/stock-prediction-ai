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
STOP_LOSS = -0.07
SLIPPAGE = 0.002
COMMISSION = 0.001
INITIAL_CAPITAL = 1.0

# =========================
# load
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

# =========================
# signal engine（そのまま使う）
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
        row["high_break_20d"] > 0 or
        row["bb_position"] > 0.7
    )

    risk_ok = (
        row["gap_up_ratio"] < 0.05 and
        row["atr_ratio"] < 0.12 and
        row["range_compression_5d"] > 0.95
    )

    if momentum and trend and volume and risk_ok:
        return 1

    elif momentum and trend and breakout:
        return 0.5

    else:
        return 0


# =========================
# backtest
# =========================
for start, end in TEST_WINDOWS:

    print(f"\n===== WINDOW {start} → {end} =====")

    d = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    # =========================
    # signal生成（ML完全なし）
    # =========================
    d["signal"] = d.apply(generate_signal, axis=1)

    d["rank"] = d.groupby("Date")["signal"].rank(ascending=False)

    # =========================
    # simple backtest
    # =========================
    dates = sorted(d["Date"].unique())

    positions = []
    trade_log = []

    for date in dates:

        new_positions = []

        # =====================
        # ポジション管理
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
        # エントリー
        # =====================
        slots = MAX_POSITIONS - len(positions)

        if slots > 0:

            picks = d[d["Date"] == date]
            picks = picks.sort_values("rank", ascending=False).head(slots)

            for _, row in picks.iterrows():

                # 仮のリターン（後で価格ベースに差し替え可）
                future_ret = row.get("return_5d", 0)

                positions.append({
                    "Ticker": row["Ticker"],
                    "entry": date,
                    "ret": future_ret,
                    "signal": row["signal"]
                })

    # =========================
    # 結果集計
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
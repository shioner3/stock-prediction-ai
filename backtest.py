import pandas as pd
import numpy as np
import pickle

# =========================
# 設定
# =========================
DATA_PATH = "stock_data/ml_dataset.parquet"
MODEL_PATH = "stock_data/lgbm_ranker.pkl"

INITIAL_CAPITAL = 1.0

TOP_N = 3
MAX_POSITIONS = 5
HOLD_DAYS = 5

STOP_LOSS = -0.07
SLIPPAGE = 0.002
COMMISSION = 0.001

# =========================
# walk-forward
# =========================
TEST_WINDOWS = [
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
]

# =========================
# features
# =========================
FEATURES = [
    "close_ma5_ratio","close_ma25_ratio","ma25_slope","high_break_20d",
    "return_5d","return_20d","relative_strength_20d","industry_rs_rank",
    "volume_ratio_5d","volume_ratio_20d","volume_zscore",
    "atr_ratio","bb_width","range_compression_5d",
    "nikkei_return_5d","topix_trend","growth_index_strength",
    "return_rank_daily","volume_rank_daily","volatility_rank","rs_rank_cross_section",
    "upper_shadow_ratio","gap_up_ratio","bb_position"
]

# =========================
# load
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# =========================
# store
# =========================
all_results = []
all_trade_logs = []

# =========================
# walk forward
# =========================
for test_start, test_end in TEST_WINDOWS:

    print(f"\n===== WINDOW {test_start} → {test_end} =====")

    df_window = df[
        (df["Date"] >= test_start) &
        (df["Date"] <= test_end)
    ].copy()

    # =====================
    # predict
    # =====================
    df_window["pred_score"] = model.predict(df_window[FEATURES])

    df_window["pred_rank"] = (
        df_window.groupby("Date")["pred_score"]
        .rank(ascending=False, method="first")
    )

    # =====================
    # filters
    # =====================
    df_window = df_window[
        (df_window["return_5d"] < 0.25) &
        (df_window["gap_up_ratio"] < 0.08) &
        (df_window["upper_shadow_ratio"] < 0.04) &
        (df_window["bb_position"] < 0.95) &
        (df_window["atr_ratio"] < 0.12) &
        (df_window["volume_ratio_20d"] > 0.3)
    ]

    df_window["target_return"] = df_window["target_return"].clip(-0.3, 0.5)

    dates = sorted(df_window["Date"].unique())

    capital = INITIAL_CAPITAL
    positions = []

    equity_curve = []
    trade_log = []

    # =====================
    # backtest
    # =====================
    for current_date in dates:

        realized_returns = []
        remaining_positions = []

        # EXIT
        for pos in positions:

            hold_days = (current_date - pos["entry_date"]).days

            if hold_days >= HOLD_DAYS:

                ret = pos["return"]
                ret = max(ret, STOP_LOSS)
                ret -= (SLIPPAGE + COMMISSION)

                realized_returns.append(ret)

                trade_log.append({
                    "Date": current_date,
                    "Ticker": pos["Ticker"],
                    "Return": ret,
                    "pred_score": pos["pred_score"],
                    "rank": pos["rank"]
                })

            else:
                remaining_positions.append(pos)

        positions = remaining_positions

        # ENTRY
        slots = MAX_POSITIONS - len(positions)

        if slots > 0:

            day_candidates = df_window[
                df_window["Date"] == current_date
            ].sort_values("pred_rank")

            for _, row in day_candidates.head(slots).iterrows():

                positions.append({
                    "Ticker": row["Ticker"],
                    "entry_date": current_date,
                    "return": row["target_return"],
                    "pred_score": row["pred_score"],
                    "rank": row["pred_rank"]
                })

        daily_ret = np.mean(realized_returns) if realized_returns else 0
        capital *= (1 + daily_ret)
        equity_curve.append(capital)

    # =========================
    # equity metrics
    # =========================
    equity = pd.Series(equity_curve)

    final_value = equity.iloc[-1]
    cagr = final_value ** (365 / len(equity)) - 1 if len(equity) > 0 else 0

    sharpe = (
        np.mean(np.diff(equity)) / np.std(np.diff(equity)) * np.sqrt(252)
        if len(equity) > 2 else 0
    )

    # =========================
    # trade df
    # =========================
    trade_df = pd.DataFrame(trade_log)

    # ==========================================================
    # ① 勝ち vs 負け差分
    # ==========================================================
    if len(trade_df) > 0:

        winners = trade_df[trade_df["Return"] > 0]
        losers = trade_df[trade_df["Return"] < 0]

        print("\n===== WIN vs LOSE DIFF (score / rank) =====")

        if len(winners) > 0 and len(losers) > 0:

            for col in ["pred_score", "rank"]:

                print(f"\n--- {col} ---")
                print("WIN mean:", winners[col].mean())
                print("LOSE mean:", losers[col].mean())
                print("DIFF:", winners[col].mean() - losers[col].mean())

    # ==========================================================
    # ② pred_score分布比較
    # ==========================================================
    if len(trade_df) > 0:

        print("\n===== PRED SCORE DISTRIBUTION =====")

        print("WIN:")
        print(winners["pred_score"].describe())

        print("\nLOSE:")
        print(losers["pred_score"].describe())

    # ==========================================================
    # ③ rank別勝率
    # ==========================================================
    if len(trade_df) > 0:

        print("\n===== RANK WIN RATE =====")

        rank_stats = trade_df.groupby("rank")["Return"].apply(
            lambda x: (x > 0).mean()
        )

        print(rank_stats)

    # =========================
    # save
    # =========================
    all_trade_logs.append(trade_df)

    all_results.append({
        "window": f"{test_start}->{test_end}",
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Final": final_value
    })

# =========================
# summary
# =========================
result_df = pd.DataFrame(all_results)

print("\n===== WALK FORWARD RESULT =====")
print(result_df)

print("\n===== AVG =====")
print(result_df.mean(numeric_only=True))
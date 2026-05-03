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
# walk-forward設定
# =========================
TEST_WINDOWS = [
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
]

# =========================
# 特徴量
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
# データ & モデル
# =========================
print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# =========================
# 結果格納
# =========================
all_results = []
all_losing_trades = []

# =========================
# WALK FORWARD LOOP
# =========================
for test_start, test_end in TEST_WINDOWS:

    print(f"\n===== WINDOW {test_start} → {test_end} =====")

    df_window = df[
        (df["Date"] >= test_start) &
        (df["Date"] <= test_end)
    ].copy()

    # =====================
    # 予測
    # =====================
    df_window["pred_score"] = model.predict(df_window[FEATURES])

    df_window["pred_rank"] = (
        df_window.groupby("Date")["pred_score"]
        .rank(ascending=False, method="first")
    )

    # =====================
    # フィルタ
    # =====================
    df_window = df_window[
        (df_window["return_5d"] < 0.25) &
        (df_window["gap_up_ratio"] < 0.08) &
        (df_window["upper_shadow_ratio"] < 0.04) &
        (df_window["bb_position"] < 0.95) &
        (df_window["atr_ratio"] < 0.12) &
        (df_window["volume_ratio_20d"] > 0.3)
    ]

    # =====================
    # target clip
    # =====================
    df_window["target_return"] = df_window["target_return"].clip(-0.3, 0.5)

    # =====================
    # バックテスト
    # =====================
    dates = sorted(df_window["Date"].unique())

    capital = INITIAL_CAPITAL
    positions = []

    equity_curve = []
    daily_returns_list = []
    trade_log = []

    for current_date in dates:

        realized_returns = []
        remaining_positions = []

        # =====================
        # EXIT
        # =====================
        for pos in positions:

            hold_days = (current_date - pos["entry_date"]).days

            if hold_days >= HOLD_DAYS:

                ret = pos["return"]
                ret = max(ret, STOP_LOSS)
                ret -= (SLIPPAGE + COMMISSION)

                realized_returns.append(ret)

                trade_log.append({
                    "Date": current_date,
                    "Return": ret,
                    "Ticker": pos["Ticker"]
                })

            else:
                remaining_positions.append(pos)

        positions = remaining_positions

        # =====================
        # ENTRY
        # =====================
        slots = MAX_POSITIONS - len(positions)

        if slots > 0:

            day_candidates = df_window[
                df_window["Date"] == current_date
            ].sort_values("pred_rank")

            for _, row in day_candidates.head(slots).iterrows():

                positions.append({
                    "Ticker": row["Ticker"],
                    "entry_date": current_date,
                    "return": row["target_return"]
                })

        # =====================
        # DAILY RETURN
        # =====================
        daily_ret = np.mean(realized_returns) if realized_returns else 0
        capital *= (1 + daily_ret)

        daily_returns_list.append(capital)

    # =========================
    # equity
    # =========================
    equity = pd.Series(daily_returns_list)

    final_value = equity.iloc[-1]

    cagr = final_value ** (365 / len(equity)) - 1 if len(equity) > 0 else 0

    sharpe = (
        np.mean(np.diff(equity)) / np.std(np.diff(equity)) * np.sqrt(252)
        if len(equity) > 2 else 0
    )

    max_dd = (equity / equity.cummax() - 1).min()

    # =========================
    # 負けトレード分析
    # =========================
    trade_df = pd.DataFrame(trade_log)

    if len(trade_df) > 0:

        losers = trade_df[trade_df["Return"] < 0].copy()

        # =====================
        # 追加：特徴量分布分析
        # =====================
        if len(losers) > 0:

            print("\n===== LOSING FEATURE ANALYSIS =====")

            # mergeして特徴量付与
            losers_full = losers.merge(
                df_window[["Date", "Ticker"] + FEATURES],
                on=["Date", "Ticker"],
                how="left"
            )

            for col in FEATURES:
                print(f"\n--- {col} ---")
                print(losers_full[col].describe())

            all_losing_trades.append({
                "window": f"{test_start}->{test_end}",
                "avg_return": losers["Return"].mean(),
                "median": losers["Return"].median(),
                "count": len(losers)
            })

    # =========================
    # 結果保存
    # =========================
    all_results.append({
        "window": f"{test_start}->{test_end}",
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "Final": final_value
    })

# =========================
# 集計
# =========================
result_df = pd.DataFrame(all_results)
loser_df = pd.DataFrame(all_losing_trades)

print("\n===== WALK FORWARD RESULT =====")
print(result_df)

print("\n===== AVG =====")
print(result_df.mean(numeric_only=True))

print("\n===== LOSING TRADE SUMMARY =====")
print(loser_df)
print("\nAVG LOSER STATS")
print(loser_df.mean(numeric_only=True))
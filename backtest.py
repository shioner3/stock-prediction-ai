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
TRAIN_START = "2018-01-01"
TEST_WINDOWS = [
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
]

# =========================
# 特徴量
# =========================
FEATURES = [
    "close_ma5_ratio",
    "close_ma25_ratio",
    "ma25_slope",
    "high_break_20d",
    "return_5d",
    "return_20d",
    "relative_strength_20d",
    "industry_rs_rank",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "volume_zscore",
    "atr_ratio",
    "bb_width",
    "range_compression_5d",
    "nikkei_return_5d",
    "topix_trend",
    "growth_index_strength",
    "return_rank_daily",
    "volume_rank_daily",
    "volatility_rank",
    "rs_rank_cross_section",
    "upper_shadow_ratio",
    "gap_up_ratio",
    "bb_position"
]

# =========================
# データ読み込み
# =========================
print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

# =========================
# モデル
# =========================
print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# =========================
# 結果格納
# =========================
all_results = []

# =========================
# walk-forward loop
# =========================
for test_start, test_end in TEST_WINDOWS:

    print(f"\n===== WINDOW {test_start} → {test_end} =====")

    # -------------------------
    # データ切り出し
    # -------------------------
    df_window = df[
        (df["Date"] >= test_start) &
        (df["Date"] <= test_end)
    ].copy()

    df_window["pred_score"] = model.predict(df_window[FEATURES])

    df_window["pred_rank"] = (
        df_window.groupby("Date")["pred_score"]
        .rank(ascending=False, method="first")
    )

    # -------------------------
    # フィルタ
    # -------------------------
    df_window = df_window[
        df_window["return_5d"] < 0.25
    ]
    df_window = df_window[
        df_window["gap_up_ratio"] < 0.08
    ]
    df_window = df_window[
        df_window["upper_shadow_ratio"] < 0.04
    ]
    df_window = df_window[
        df_window["bb_position"] < 0.95
    ]
    df_window = df_window[
        df_window["atr_ratio"] < 0.12
    ]
    df_window = df_window[
        df_window["volume_ratio_20d"] > 0.3
    ]

    # -------------------------
    # clip（重要）
    # -------------------------
    df_window["target_return"] = (
        df_window["target_return"].clip(-0.3, 0.5)
    )

    # =========================
    # バックテスト
    # =========================
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
        # ポジション決済
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
                    "Return": ret
                })

            else:
                remaining_positions.append(pos)

        positions = remaining_positions

        # =====================
        # エントリー
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
        # 日次収益
        # =====================
        daily_ret = np.mean(realized_returns) if len(realized_returns) > 0 else 0

        capital *= (1 + daily_ret)

        daily_returns_list.append(capital)

    # =========================
    # 集計
    # =========================
    equity_curve = pd.Series(daily_returns_list)

    final_value = equity_curve.iloc[-1]

    cagr = final_value ** (365 / len(equity_curve)) - 1

    sharpe = np.mean(np.diff(equity_curve)) / np.std(np.diff(equity_curve)) * np.sqrt(252) if len(equity_curve) > 1 else 0

    max_dd = (equity_curve / equity_curve.cummax() - 1).min()

    all_results.append({
        "window": f"{test_start}->{test_end}",
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "Final": final_value
    })

# =========================
# 結果
# =========================
result_df = pd.DataFrame(all_results)

print("\n===== WALK FORWARD RESULT =====")
print(result_df)

print("\n===== AVG =====")
print(result_df.mean(numeric_only=True))
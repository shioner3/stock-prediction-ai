import pandas as pd
import numpy as np
import pickle

# =========================
# 設定
# =========================
DATA_PATH = "stock_data/ml_dataset.parquet"

FEATURES = [
    "close_ma5_ratio","close_ma25_ratio","ma25_slope","high_break_20d",
    "return_5d","return_20d","relative_strength_20d","industry_rs_rank",
    "volume_ratio_5d","volume_ratio_20d","volume_zscore",
    "atr_ratio","bb_width","range_compression_5d",
    "nikkei_return_5d","topix_trend","growth_index_strength",
    "return_rank_daily","volume_rank_daily","volatility_rank","rs_rank_cross_section",
    "upper_shadow_ratio","gap_up_ratio","bb_position"
]

TEST_WINDOWS = [
    ("2022-01-01","2022-12-31"),
    ("2023-01-01","2023-12-31"),
    ("2024-01-01","2024-12-31"),
]

TOP_N = 3
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
df = df.sort_values(["Date","Ticker"])

with open(MODEL_PATH,"rb") as f:
    model = pickle.load(f)

# =========================
# 全結果格納
# =========================
all_feature_stats = []

# =========================
# WALK FORWARD
# =========================
for start,end in TEST_WINDOWS:

    print(f"\n===== WINDOW {start} → {end} =====")

    d = df[(df["Date"]>=start)&(df["Date"]<=end)].copy()

    # =====================
    # prediction
    # =====================
    d["pred"] = model.predict(d[FEATURES])
    d["rank"] = d.groupby("Date")["pred"].rank(ascending=False)

    # =====================
    # simple backtest
    # =====================
    dates = sorted(d["Date"].unique())

    positions = []
    trade_log = []

    for date in dates:

        realized = []
        new_positions = []

        for p in positions:

            hold = (date - p["entry"]).days

            if hold >= HOLD_DAYS:

                ret = max(p["ret"], STOP_LOSS)
                ret -= (SLIPPAGE + COMMISSION)

                realized.append(ret)

                trade_log.append({
                    "Date": date,
                    "Ticker": p["Ticker"],
                    "Return": ret
                })

            else:
                new_positions.append(p)

        positions = new_positions

        slots = MAX_POSITIONS - len(positions)

        if slots > 0:

            picks = d[d["Date"]==date].sort_values("rank").head(slots)

            for _, row in picks.iterrows():

                positions.append({
                    "Ticker": row["Ticker"],
                    "entry": date,
                    "ret": row["target_return"]
                })

    # =====================
    # merge feature
    # =====================
    trade_df = pd.DataFrame(trade_log)

    if len(trade_df) == 0:
        continue

    trade_df = trade_df.merge(
        d[["Date","Ticker"]+FEATURES],
        on=["Date","Ticker"],
        how="left"
    )

    # =====================
    # split win / lose
    # =====================
    wins = trade_df[trade_df["Return"] > 0]
    loses = trade_df[trade_df["Return"] <= 0]

    print("\n===== WIN vs LOSE FEATURE DIFFERENCE =====")

    feature_diff = []

    for col in FEATURES:

        win_mean = wins[col].mean()
        lose_mean = loses[col].mean()

        diff = win_mean - lose_mean

        # z-score差分（標準化）
        pooled_std = trade_df[col].std()

        z_diff = diff / pooled_std if pooled_std != 0 else 0

        feature_diff.append({
            "feature": col,
            "win_mean": win_mean,
            "lose_mean": lose_mean,
            "diff": diff,
            "z_diff": z_diff
        })

        print(f"\n--- {col} ---")
        print(f"WIN : {win_mean:.4f}")
        print(f"LOSE: {lose_mean:.4f}")
        print(f"DIFF: {diff:.4f}")
        print(f"Z-DIFF: {z_diff:.4f}")

    feature_df = pd.DataFrame(feature_diff)

    # =====================
    # ranking（重要）
    # =====================
    feature_df = feature_df.sort_values("z_diff", ascending=False)

    print("\n===== TOP POSITIVE SIGNALS =====")
    print(feature_df.head(10))

    print("\n===== NEGATIVE SIGNALS =====")
    print(feature_df.tail(10))

    # =====================
    # store
    # =====================
    all_feature_stats.append(feature_df)

# =========================
# 全体統合
# =========================
full_df = pd.concat(all_feature_stats)

summary = full_df.groupby("feature").agg({
    "z_diff": ["mean","std"],
    "diff": "mean"
})

print("\n===== GLOBAL SIGNAL SUMMARY =====")
print(summary.sort_values(("z_diff","mean"), ascending=False))
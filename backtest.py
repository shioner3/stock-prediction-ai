import pandas as pd
import numpy as np
import pickle

# =========================
# 設定
# =========================
DATA_PATH = "stock_data/ml_dataset.parquet"
MODEL_PATH = "stock_data/lgbm_ranker.pkl"

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

# =========================
# load
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date","Ticker"])

with open(MODEL_PATH,"rb") as f:
    model = pickle.load(f)

all_ic = []

# =========================
# walk forward
# =========================
for start,end in TEST_WINDOWS:

    print(f"\n===== {start} → {end} =====")

    d = df[(df["Date"]>=start)&(df["Date"]<=end)].copy()

    # =====================
    # predict
    # =====================
    d["pred"] = model.predict(d[FEATURES])

    # =====================
    # IC計算（日次cross-section）
    # =====================
    ic_list = []

    for date, g in d.groupby("Date"):

        if len(g) < 5:
            continue

        ic = g["pred"].corr(g["target_return"])
        ic_list.append(ic)

    ic_series = pd.Series(ic_list).dropna()

    # =====================
    # IC統計
    # =====================
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    icir = mean_ic / std_ic if std_ic != 0 else 0

    print("\n===== IC RESULT =====")
    print(f"Mean IC : {mean_ic:.6f}")
    print(f"IC STD  : {std_ic:.6f}")
    print(f"ICIR    : {icir:.6f}")
    print(f"Hit Rate (IC>0): {(ic_series>0).mean():.3f}")

    # =====================
    # 月次IC
    # =====================
    d["IC_date"] = d["Date"].dt.to_period("M")

    monthly_ic = d.groupby("IC_date").apply(
        lambda x: x["pred"].corr(x["target_return"])
    )

    print("\n===== MONTHLY IC =====")
    print(monthly_ic)

    # =====================
    # 保存
    # =====================
    all_ic.append({
        "window": f"{start}->{end}",
        "mean_ic": mean_ic,
        "ic_std": std_ic,
        "icir": icir,
        "hit_rate": (ic_series>0).mean()
    })

# =========================
# summary
# =========================
ic_df = pd.DataFrame(all_ic)

print("\n===== IC SUMMARY =====")
print(ic_df)

print("\n===== AVG IC =====")
print(ic_df.mean(numeric_only=True))
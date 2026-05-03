import pandas as pd
import numpy as np
import pickle

# =========================
# 設定
# =========================
FEATURE_PATH = "stock_data/technical_features.parquet"

MODEL_PATH = "stock_data/lgbm_ranker.pkl"

SAVE_PATH = "stock_data/predictions.csv"

TOP_N = 10

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
    "turnover_ratio",
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
print("Loading features...")

df = pd.read_parquet(FEATURE_PATH)

df["Date"] = pd.to_datetime(df["Date"])

# =========================
# 最新日取得
# =========================
latest_date = df["Date"].max()

latest_df = df[
    df["Date"] == latest_date
].copy()

print(f"Prediction Date: {latest_date}")

# =========================
# 欠損除去
# =========================
latest_df = latest_df.dropna(
    subset=FEATURES
)

# =========================
# モデル読み込み
# =========================
print("Loading model...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# =========================
# prediction
# =========================
print("Predicting...")

latest_df["pred_score"] = model.predict(
    latest_df[FEATURES]
)

# =========================
# rank
# =========================
latest_df = latest_df.sort_values(
    "pred_score",
    ascending=False
)

latest_df["rank"] = np.arange(
    1,
    len(latest_df) + 1
)

# =========================
# 過熱除外フィルタ
# =========================
latest_df = latest_df[
    latest_df["return_5d"] < 0.30
]

latest_df = latest_df[
    latest_df["gap_up_ratio"] < 0.10
]

latest_df = latest_df[
    latest_df["upper_shadow_ratio"] < 0.05
]

latest_df = latest_df[
    latest_df["bb_position"] < 0.98
]

latest_df = latest_df[
    latest_df["atr_ratio"] < 0.15
]

# =========================
# TOP N
# =========================
result_df = latest_df.head(TOP_N)

# =========================
# 表示列
# =========================
display_cols = [

    "rank",
    "Ticker",
    "pred_score",

    "return_5d",
    "return_20d",

    "volume_zscore",

    "atr_ratio",

    "bb_position",

    "gap_up_ratio",

    "upper_shadow_ratio"
]

# 存在する列のみ
display_cols = [
    col for col in display_cols
    if col in result_df.columns
]

# =========================
# 保存
# =========================
print("\n=== TOP PICKS ===")

print(
    result_df[display_cols]
)

result_df[display_cols].to_csv(
    SAVE_PATH,
    index=False
)

print(f"\nSaved: {SAVE_PATH}")
print("Done.")
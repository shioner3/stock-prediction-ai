import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle

from lightgbm import LGBMRanker
from sklearn.metrics import ndcg_score

# =========================
# 設定
# =========================
DATA_PATH = "stock_data/ml_dataset.parquet"

MODEL_SAVE_PATH = "stock_data/lgbm_ranker.pkl"

TARGET_COLUMN = "target_rank"

TRAIN_END_DATE = "2022-12-31"
VALID_END_DATE = "2023-12-31"

TOP_N = 3

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

df = df.sort_values(["Date", "Ticker"])

# =========================
# train / valid / test split
# =========================
train_df = df[
    df["Date"] <= TRAIN_END_DATE
]

valid_df = df[
    (df["Date"] > TRAIN_END_DATE)
    &
    (df["Date"] <= VALID_END_DATE)
]

test_df = df[
    df["Date"] > VALID_END_DATE
]

print(f"Train size: {len(train_df)}")
print(f"Valid size: {len(valid_df)}")
print(f"Test size : {len(test_df)}")

# =========================
# group 作成
# =========================
train_group = (
    train_df.groupby("Date")
    .size()
    .values
)

valid_group = (
    valid_df.groupby("Date")
    .size()
    .values
)

test_group = (
    test_df.groupby("Date")
    .size()
    .values
)

# =========================
# X / y
# =========================
X_train = train_df[FEATURES]
y_train = train_df[TARGET_COLUMN]

X_valid = valid_df[FEATURES]
y_valid = valid_df[TARGET_COLUMN]

X_test = test_df[FEATURES]
y_test = test_df[TARGET_COLUMN]

# =========================
# モデル
# =========================
print("Training LGBMRanker...")

model = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",

    boosting_type="gbdt",

    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,

    n_estimators=300,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,
    n_jobs=-1
)

# =========================
# 学習
# =========================
model.fit(
    X_train,
    y_train,

    group=train_group,

    eval_set=[(X_valid, y_valid)],
    eval_group=[valid_group],

    eval_at=[TOP_N]
)

# =========================
# importance
# =========================
print("\n=== Feature Importance ===")

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print(importance_df)

# =========================
# prediction
# =========================
print("\nPredicting test set...")

test_df["pred_score"] = model.predict(X_test)

# =========================
# 日次rank
# =========================
test_df["pred_rank"] = (
    test_df.groupby("Date")["pred_score"]
    .rank(ascending=False)
)

# =========================
# TOP N精度確認
# =========================
print(f"\n=== TOP {TOP_N} Average Return ===")

top_df = test_df[
    test_df["pred_rank"] <= TOP_N
]

daily_return = (
    top_df.groupby("Date")["target_return"]
    .mean()
)

print(daily_return.describe())

# =========================
# NDCG確認
# =========================
print("\nCalculating NDCG...")

ndcg_list = []

for date, group in test_df.groupby("Date"):

    true_relevance = [group["target_rank"].values]
    pred_relevance = [group["pred_score"].values]

    try:
        score = ndcg_score(
            true_relevance,
            pred_relevance,
            k=TOP_N
        )

        ndcg_list.append(score)

    except:
        pass

print(f"NDCG@{TOP_N}: {np.mean(ndcg_list):.4f}")

# =========================
# モデル保存
# =========================
print("\nSaving model...")

with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump(model, f)

print("Done.")
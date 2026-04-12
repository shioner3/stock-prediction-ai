import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRanker

# =========================
# 設定
# =========================
TOP_N = 3
CANDIDATE_N = 10
N_CLASS = 30
DIVERSITY_BUCKETS = 3

BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest.parquet")

# =========================
# データ読み込み
# =========================
train_df = pd.read_parquet(TRAIN_DATA_PATH)
predict_df = pd.read_parquet(PREDICT_DATA_PATH)

# =========================
# FEATURES
# =========================
FEATURES = [
    "Return_1","Return_3",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio",
    "HL_range",
    "Rel_Return_1",
    "Trend_5_z","Trend_10_z","Trend_diff",
    "Gap","Volatility_change","Momentum_acc",
    "DD_5","DD_10",
    "TrendVol","Volume_Z",
    "Return_1_rank","Volume_ratio_rank",
    "Trend_5_z_rank","TrendVol_rank",
    "Market_Z","Market_Trend"
]

# =========================
# 前処理
# =========================
train_df = train_df.dropna(subset=FEATURES + ["Target"]).copy()
predict_df = predict_df.dropna(subset=FEATURES).copy()

# =========================
# Ranker用Target
# =========================
train_df["TargetRank"] = train_df.groupby("Date")["Target"].transform(
    lambda x: pd.qcut(x, q=N_CLASS, labels=False, duplicates="drop")
)

train_df = train_df.dropna(subset=["TargetRank"])
train_df["TargetRank"] = train_df["TargetRank"].astype(int)

train_df = train_df.sort_values("Date")
group = train_df.groupby("Date").size().tolist()

assert train_df["TargetRank"].max() < N_CLASS

# =========================
# モデル
# =========================
model = LGBMRanker(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    train_df[FEATURES],
    train_df["TargetRank"],
    group=group
)

# =========================
# 予測
# =========================
today = predict_df.copy()
today["score_raw"] = model.predict(today[FEATURES])

# =========================
# ① FILTER（弱め）
# =========================
today = today[
    today["TrendVol"] > -1.0
].copy()

# =========================
# ② SCORE
# =========================
today["score"] = today["score_raw"]

# =========================
# ③ TOP10候補
# =========================
candidates = today.sort_values("score", ascending=False).head(CANDIDATE_N).copy()

# =========================
# ④ DIVERSITY（本質修正）
# bucket → 各bucketから1銘柄ずつ選ぶ
# =========================

# 全体でbucket作成（←ここが重要）
candidates["vol_bucket"] = pd.qcut(
    candidates["TrendVol"],
    q=min(DIVERSITY_BUCKETS, len(candidates)),
    labels=False,
    duplicates="drop"
)

selected = []

# bucketごとに1銘柄選択
for b in sorted(candidates["vol_bucket"].dropna().unique()):
    tmp = candidates[candidates["vol_bucket"] == b]
    if len(tmp) > 0:
        best = tmp.sort_values("score", ascending=False).iloc[0]
        selected.append(best)

selected = pd.DataFrame(selected)

# =========================
# 不足分補充（重要）
# =========================
if len(selected) < TOP_N:
    remain = candidates[~candidates.index.isin(selected.index)]
    remain = remain.sort_values("score", ascending=False)
    selected = pd.concat([selected, remain.head(TOP_N - len(selected))])

# =========================
# ⑤ TOP3最終選定
# =========================
final = selected.head(TOP_N).copy()
final["rank"] = range(1, len(final) + 1)

# =========================
# weight
# =========================
final["weight"] = np.exp(final["score"])
final["weight"] /= final["weight"].sum()

# =========================
# 出力
# =========================
print("\n=== 今日の銘柄（FINAL） ===")

print(final[[
    "Ticker",
    "score",
    "Trend_5_z",
    "TrendVol",
    "DD_5",
    "vol_bucket",
    "weight",
    "rank"
]])

print("\n件数:", len(final))
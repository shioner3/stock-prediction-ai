import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRanker

# =========================
# 設定（15日専用）
# =========================
TOP_N = 3
CANDIDATE_N = 10
N_CLASS = 30
DIVERSITY_BUCKETS = 3

# 🔥 15日用パラ（後で最適化可能）
W_TRENDVOL = 0.6
W_DD = 0.3

BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_15d.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest_15d.parquet")

# =========================
# データ読み込み
# =========================
train_df = pd.read_parquet(TRAIN_DATA_PATH)
predict_df = pd.read_parquet(PREDICT_DATA_PATH)

# =========================
# FEATURES（15日専用）
# =========================
FEATURES = [
    "Return_5","Return_10","Return_20",
    "MA5_ratio","MA10_ratio","MA20_ratio","MA30_ratio",
    "Volatility",
    "Trend_10_z","Trend_20_z","Trend_40_z",
    "DD_20","DD_40",
    "TrendVol","Volume_Z",
    "Return_10_rank","Trend_20_z_rank",
    "TrendVol_rank","DD_20_rank",
    "Market_Z","Market_Trend"
]

# =========================
# 前処理
# =========================
train_df = train_df.dropna(subset=FEATURES + ["Target"]).copy()
predict_df = predict_df.dropna(subset=FEATURES).copy()

# =========================
# Ranker target
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
    random_state=42,
    n_jobs=-1
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
# 🔥 final_score（15日の核）
# =========================
today["trend_rank"] = today["TrendVol"].rank(pct=True)
today["dd_rank"] = (-today["DD_20"]).rank(pct=True)  # 押し目はプラス評価

today["filter_score"] = (
    W_TRENDVOL * today["trend_rank"] +
    W_DD * today["dd_rank"]
)

today["score"] = today["score_raw"] * (1 + today["filter_score"])

# =========================
# ① FILTER（ほぼ無し）
# =========================
# → 長期は基本フィルタ弱く
today = today[today["TrendVol"].notna()].copy()

# =========================
# ② TOP候補
# =========================
candidates = today.sort_values("score", ascending=False).head(CANDIDATE_N).copy()

# =========================
# ③ DIVERSITY（TrendVol分散）
# =========================
candidates["vol_bucket"] = pd.qcut(
    candidates["TrendVol"],
    q=min(DIVERSITY_BUCKETS, len(candidates)),
    labels=False,
    duplicates="drop"
)

selected = []

for b in sorted(candidates["vol_bucket"].dropna().unique()):
    tmp = candidates[candidates["vol_bucket"] == b]
    if len(tmp) > 0:
        best = tmp.sort_values("score", ascending=False).iloc[0]
        selected.append(best)

selected = pd.DataFrame(selected)

# =========================
# 不足補充
# =========================
if len(selected) < TOP_N:
    remain = candidates[~candidates.index.isin(selected.index)]
    remain = remain.sort_values("score", ascending=False)
    selected = pd.concat([selected, remain.head(TOP_N - len(selected))])

# =========================
# 最終
# =========================
final = selected.head(TOP_N).copy()
final["rank"] = range(1, len(final) + 1)

# =========================
# weight（長期は均等寄りでもOK）
# =========================
final["weight"] = np.exp(final["score"])
final["weight"] /= final["weight"].sum()

# =========================
# 出力
# =========================
print("\n=== 今日の銘柄（15日モデル） ===")

print(final[[
    "Ticker",
    "score",
    "TrendVol",
    "DD_20",
    "vol_bucket",
    "weight",
    "rank"
]])

print("\n件数:", len(final))
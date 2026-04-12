import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRanker

# =========================
# 設定
# =========================
TOP_N = 3
CANDIDATE_N = 10   # ← TOP10生成用
N_CLASS = 30

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
# Ranker用Target（qcut）
# =========================
train_df["TargetRank"] = train_df.groupby("Date")["Target"].transform(
    lambda x: pd.qcut(x, q=N_CLASS, labels=False, duplicates="drop")
)

train_df = train_df.dropna(subset=["TargetRank"])
train_df["TargetRank"] = train_df["TargetRank"].astype(int)

train_df = train_df.sort_values("Date")
group = train_df.groupby("Date").size().tolist()

# safety check
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
# ① FILTER（戦略制約）
# =========================
today = today[
    (today["Trend_5_z"] > 0) &
    (today["TrendVol"] > 0)
].copy()

# =========================
# ② SCORE
# =========================
today["score"] = today["score_raw"]

# =========================
# ③ TOP10候補抽出
# =========================
candidates = today.sort_values("score", ascending=False).head(CANDIDATE_N).copy()

# =========================
# ④ DIVERSITY制御
# （セクター無し簡易版：銘柄バラけさせる）
# =========================

# score順にしつつ、同一銘柄偏り回避（簡易版）
candidates = candidates.sort_values(
    ["Trend_5_z", "score"],
    ascending=[False, False]
)

# 上位からユニーク優先（今回はTicker重複は基本ないが保険）
candidates = candidates.drop_duplicates(subset=["Ticker"])

# =========================
# ⑤ TOP3最終選定
# =========================
final = candidates.head(TOP_N).copy()

final["rank"] = range(1, len(final) + 1)

# =========================
# weight（安定版）
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
    "weight",
    "rank"
]])

print("\n件数:", len(final))
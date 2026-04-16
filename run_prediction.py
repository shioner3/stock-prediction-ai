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

# 🔥 重み（安定版）
W_TRENDVOL = 0.3
W_DD = 0.4
W_MOM = 0.3

BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_15d.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest_15d.parquet")

# =========================
# データ
# =========================
train_df = pd.read_parquet(TRAIN_DATA_PATH)
predict_df = pd.read_parquet(PREDICT_DATA_PATH)

FEATURES = [
    "Return_5","Return_10","Return_20","Momentum_20",
    "MA5_ratio","MA10_ratio","MA20_ratio","MA30_ratio",
    "Volatility",
    "Trend_10_z","Trend_20_z","Trend_40_z",
    "DD_20","DD_40",
    "TrendVol","Volume_Z",
    "Return_10_rank","Trend_20_z_rank",
    "TrendVol_rank","DD_20_rank",
    "Market_Z","Market_Trend",
    "Market_Vol","Market_Trend_Str"
]

# =========================
# 安全チェック
# =========================
missing = [c for c in FEATURES if c not in train_df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# =========================
# 前処理
# =========================
train_df = train_df.dropna(subset=FEATURES + ["Target"]).copy()
predict_df = predict_df.dropna(subset=FEATURES).copy()

# =========================
# Ranker
# =========================
train_df["TargetRank"] = train_df.groupby("Date")["Target"].transform(
    lambda x: pd.qcut(x, q=N_CLASS, labels=False, duplicates="drop")
)

train_df = train_df.dropna(subset=["TargetRank"])
train_df["TargetRank"] = train_df["TargetRank"].astype(int)

group = train_df.groupby("Date").size().tolist()

model = LGBMRanker(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(train_df[FEATURES], train_df["TargetRank"], group=group)

# =========================
# 予測
# =========================
today = predict_df.copy()
today["score_raw"] = model.predict(today[FEATURES])

# 🔥 score標準化（超重要）
today["score_raw"] = (
    today["score_raw"] - today["score_raw"].mean()
) / (today["score_raw"].std() + 1e-9)

# =========================
# スコア（加算型に変更）
# =========================
today["trend_rank"] = today["TrendVol"].rank(pct=True)
today["dd_rank"] = (-today["DD_20"]).rank(pct=True)
today["mom_rank"] = today["Momentum_20"].rank(pct=True)

today["final_score"] = (
    today["score_raw"]
    + W_TRENDVOL * today["trend_rank"]
    + W_DD * today["dd_rank"]
    + W_MOM * today["mom_rank"]
)

# =========================
# 市場フィルタ（ゲート型）
# =========================
market_trend = today["Market_Trend"].iloc[0]
market_vol = today["Market_Vol"].iloc[0]

if market_trend < -0.02:
    print("⚠️ 地合い悪化 → ノーポジ")
    today = today.iloc[0:0]  # 全スキップ

elif market_vol > today["Market_Vol"].quantile(0.9):
    print("⚠️ ボラ過大 → 控えめ運用")

# =========================
# 候補
# =========================
candidates = today.sort_values("final_score", ascending=False).head(CANDIDATE_N).copy()

if len(candidates) == 0:
    print("⚠️ 候補なし → スキップ")
    exit()

# =========================
# diversity
# =========================
candidates["bucket"] = pd.qcut(
    candidates["TrendVol"],
    q=min(DIVERSITY_BUCKETS, len(candidates)),
    labels=False,
    duplicates="drop"
)

selected = []

for b in sorted(candidates["bucket"].dropna().unique()):
    tmp = candidates[candidates["bucket"] == b]
    if len(tmp) > 0:
        selected.append(tmp.head(1))

# fallback
if len(selected) == 0:
    print("⚠️ selected empty → fallback")
    selected = candidates.head(TOP_N)
else:
    selected = pd.concat(selected)

# =========================
# 補充
# =========================
if len(selected) < TOP_N:
    remain = candidates[~candidates.index.isin(selected.index)]
    selected = pd.concat([selected, remain.head(TOP_N - len(selected))])

# =========================
# 最終
# =========================
final = selected.head(TOP_N).copy()
final["rank"] = range(1, len(final) + 1)

# =========================
# weight（安定化）
# =========================
final["weight"] = np.exp(final["final_score"])
final["weight"] /= final["weight"].sum()

# =========================
# 出力
# =========================
print("\n=== 今日の銘柄（15日モデル・改良版） ===")

print(final[[
    "Ticker",
    "final_score",
    "TrendVol",
    "Momentum_20",
    "DD_20",
    "weight",
    "rank"
]])
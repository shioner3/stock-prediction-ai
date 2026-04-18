import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRanker

# =========================
# 設定（7日モデル）
# =========================
TOP_N = 3
CANDIDATE_N = 10
N_CLASS = 30
DIVERSITY_BUCKETS = 3

# 🔥 重み（最適化）
W_TRENDVOL = 0.25
W_DD = 0.35
W_MOM = 0.25
W_ACCEL = 0.15   # ←追加（超重要）

BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_7d.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest_7d.parquet")

# =========================
# データ
# =========================
train_df = pd.read_parquet(TRAIN_DATA_PATH)
predict_df = pd.read_parquet(PREDICT_DATA_PATH)

FEATURES = [
    "Return_5","Return_10","Return_20",
    "Momentum_20","Momentum_accel",
    "MA5_ratio","MA10_ratio","MA20_ratio","MA30_ratio",
    "Volatility",
    "Trend_10_z","Trend_20_z","Trend_40_z",
    "DD_20","DD_40",
    "TrendVol","Volume_Z",
    "Return_10_rank","Trend_20_z_rank",
    "TrendVol_rank","DD_20_rank",
    "Market_Z","Market_Trend",
    "Market_Vol","Market_Trend_Str","Market_Sharpe"
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

# =========================
# 🔥 score標準化
# =========================
today["score_raw"] = (
    today["score_raw"] - today["score_raw"].mean()
) / (today["score_raw"].std() + 1e-9)

# =========================
# 🔥 ランク特徴
# =========================
today["trend_rank"] = today["TrendVol"].rank(pct=True)
today["dd_rank"] = (-today["DD_20"]).rank(pct=True)
today["mom_rank"] = today["Momentum_20"].rank(pct=True)
today["accel_rank"] = today["Momentum_accel"].rank(pct=True)

# =========================
# 🔥 スコア（加算型・完成形）
# =========================
today["final_score"] = (
    today["score_raw"]
    + W_TRENDVOL * today["trend_rank"]
    + W_DD * today["dd_rank"]
    + W_MOM * today["mom_rank"]
    + W_ACCEL * today["accel_rank"]
)

# =========================
# 🔥 市場レジーム判定（強化版）
# =========================
market_trend = today["Market_Trend"].iloc[0]
market_vol = today["Market_Vol"].iloc[0]
market_trend_str = today["Market_Trend_Str"].iloc[0]
market_sharpe = today["Market_Sharpe"].iloc[0]

TOP_N_DYNAMIC = TOP_N

if market_trend < -0.02 or market_sharpe < -0.5:
    print("⚠️ 弱い相場 → 1銘柄")
    TOP_N_DYNAMIC = 1

elif market_trend > 0.02 and market_trend_str > 0.03 and market_sharpe > 0.5:
    print("🔥 強い相場 → フルポジ")
    TOP_N_DYNAMIC = 5

else:
    TOP_N_DYNAMIC = 3

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
    selected = candidates.head(TOP_N_DYNAMIC)
else:
    selected = pd.concat(selected)

# =========================
# 補充
# =========================
if len(selected) < TOP_N_DYNAMIC:
    remain = candidates[~candidates.index.isin(selected.index)]
    selected = pd.concat([selected, remain.head(TOP_N_DYNAMIC - len(selected))])

# =========================
# 最終
# =========================
final = selected.head(TOP_N_DYNAMIC).copy()
final["rank"] = range(1, len(final) + 1)

# =========================
# weight（安定化）
# =========================
final["weight"] = np.exp(final["final_score"])
final["weight"] /= final["weight"].sum()

# =========================
# 出力
# =========================
print("\n=== 今日の銘柄（7日モデル・完成版） ===")

print(final[[
    "Ticker",
    "final_score",
    "TrendVol",
    "Momentum_20",
    "Momentum_accel",
    "DD_20",
    "weight",
    "rank"
]])
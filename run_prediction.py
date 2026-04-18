import pandas as pd
import numpy as np
import os

# =========================
# 設定（7日モデル・スコア特化）
# =========================
TOP_N = 3
CANDIDATE_N = 10
DIVERSITY_BUCKETS = 3

BASE_DIR = os.path.dirname(__file__)

PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest_7d.parquet")

# =========================
# データ
# =========================
df = pd.read_parquet(PREDICT_DATA_PATH).copy()

# =========================
# 🔥 スコア構築（主役）
# =========================

# トレンド × モメンタム
df["Score_TrendMomentum"] = df["Trend_20_z"] * df["Momentum_20"]

# 質（上昇効率）
df["Score_Quality"] = (
    df["TrendVol_rank"]
    + (-df["DD_20_rank"])
)

# 短期モメンタム
df["Score_ShortTerm"] = (
    0.5 * df["Return_5"]
    + 0.5 * df["Return_10"]
)

# 逆張り（過熱対策）
df["Score_Reversal"] = -df["Return_5"]

# =========================
# 🔥 正規化（超重要）
# =========================
for col in [
    "Score_TrendMomentum",
    "Score_Quality",
    "Score_ShortTerm",
    "Score_Reversal"
]:
    df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)

# =========================
# 🔥 最終スコア（ここが戦略の核）
# =========================
df["final_score"] = (
    0.35 * df["Score_TrendMomentum"]
    + 0.30 * df["Score_Quality"]
    + 0.20 * df["Score_ShortTerm"]
    + 0.15 * df["Score_Reversal"]
)

# =========================
# 🔥 市場レジーム
# =========================
market_trend = df["Market_Trend"].iloc[0]
market_sharpe = df["Market_Sharpe"].iloc[0]

if market_trend < 0:
    print("⚠️ 弱気相場 → 守り")
    w_trend = 0.2
    w_rev   = 0.3
else:
    print("🔥 強気相場 → 攻め")
    w_trend = 0.4
    w_rev   = 0.1

# レジーム反映
df["final_score"] = (
    w_trend * df["Score_TrendMomentum"]
    + 0.30 * df["Score_Quality"]
    + 0.20 * df["Score_ShortTerm"]
    + w_rev * df["Score_Reversal"]
)

# =========================
# 候補抽出
# =========================
candidates = df.sort_values("final_score", ascending=False).head(CANDIDATE_N).copy()

if len(candidates) == 0:
    print("⚠️ 候補なし")
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

if len(selected) == 0:
    selected = candidates.head(TOP_N)
else:
    selected = pd.concat(selected)

# 補充
if len(selected) < TOP_N:
    remain = candidates[~candidates.index.isin(selected.index)]
    selected = pd.concat([selected, remain.head(TOP_N - len(selected))])

# =========================
# 最終
# =========================
final = selected.head(TOP_N).copy()
final["rank"] = range(1, len(final) + 1)

# =========================
# weight
# =========================
final["weight"] = np.exp(final["final_score"])
final["weight"] /= final["weight"].sum()

# =========================
# 出力
# =========================
print("\n=== 今日の銘柄（スコア戦略・完成版） ===")

print(final[[
    "Ticker",
    "final_score",
    "Score_TrendMomentum",
    "Score_Quality",
    "Score_ShortTerm",
    "Score_Reversal",
    "weight",
    "rank"
]])
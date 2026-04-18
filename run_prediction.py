import pandas as pd
import numpy as np
import os

# =========================
# 設定（爆発検出モデル）
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
# 🔥 正規化（超重要）
# =========================
for col in ["Breakout", "Volume_Spike", "Vol_Expansion", "Gap"]:
    df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)

# =========================
# 🔥 最終スコア（爆発特化）
# =========================
df["final_score"] = (
    0.40 * df["Breakout"]
    + 0.25 * df["Volume_Spike"]
    + 0.20 * df["Vol_Expansion"]
    + 0.15 * df["Gap"]
)

# =========================
# 🔥 市場フィルター（簡略版）
# =========================
market_trend = df["Market_Trend"].iloc[0]
market_sharpe = df["Market_Sharpe"].iloc[0]

if market_trend < 0 or market_sharpe < 0:
    print("⚠️ 弱い相場 → 厳選")
    CANDIDATE_N = 5
else:
    print("🔥 通常相場")

# =========================
# 候補抽出
# =========================
candidates = df.sort_values("final_score", ascending=False).head(CANDIDATE_N).copy()

if len(candidates) == 0:
    print("⚠️ 候補なし")
    exit()

# =========================
# 🔥 diversity（ボラで分散）
# =========================
candidates["bucket"] = pd.qcut(
    candidates["Vol_Expansion"],
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
# 🔥 weight（爆発対応）
# =========================
final["weight"] = np.exp(final["final_score"])
final["weight"] /= final["weight"].sum()

# =========================
# 出力
# =========================
print("\n=== 今日の銘柄（爆発検出モデル） ===")

print(final[[
    "Ticker",
    "final_score",
    "Breakout",
    "Volume_Spike",
    "Vol_Expansion",
    "Gap",
    "weight",
    "rank"
]])
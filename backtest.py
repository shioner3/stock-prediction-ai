import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRanker

# =========================
# 設定
# =========================
TOP_N = 3
CANDIDATE_N = 10
MAX_POSITIONS = 5
HOLD_DAYS = 3
N_CLASS = 30
DIVERSITY_BUCKETS = 3

DATA_PATH = "ml_dataset.parquet"

INITIAL_CAPITAL = 1.0

# =========================
# データ読み込み
# =========================
df = pd.read_parquet(DATA_PATH)

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
df = df.dropna(subset=FEATURES + ["Target"]).copy()
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

# =========================
# Ranker用Target
# =========================
df["TargetRank"] = df.groupby("Date")["Target"].transform(
    lambda x: pd.qcut(x, q=N_CLASS, labels=False, duplicates="drop")
)

df = df.dropna(subset=["TargetRank"])
df["TargetRank"] = df["TargetRank"].astype(int)

# =========================
# 学習
# =========================
train_df = df.copy()

group = train_df.groupby("Date").size().tolist()

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
# バックテスト
# =========================
dates = sorted(df["Date"].unique())

capital = INITIAL_CAPITAL
equity_curve = []

positions = []  # 現在保有

for i in range(len(dates) - HOLD_DAYS - 1):

    today = dates[i]
    next_day = dates[i + 1]

    today_df = df[df["Date"] == today].copy()

    # =========================
    # ① 予測
    # =========================
    today_df["score"] = model.predict(today_df[FEATURES])

    # =========================
    # ② フィルタ（弱め）
    # =========================
    today_df = today_df[today_df["TrendVol"] > -1.0]

    if len(today_df) == 0:
        equity_curve.append(capital)
        continue

    # =========================
    # ③ TOP候補
    # =========================
    candidates = today_df.sort_values("score", ascending=False).head(CANDIDATE_N)

    # =========================
    # ④ diversity（bucket）
    # =========================
    candidates["vol_bucket"] = pd.qcut(
        candidates["TrendVol"],
        q=min(DIVERSITY_BUCKETS, len(candidates)),
        labels=False,
        duplicates="drop"
    )

    selected = []

    for b in sorted(candidates["vol_bucket"].dropna().unique()):
        group_df = candidates[candidates["vol_bucket"] == b]
        pick = group_df.sort_values("score", ascending=False).head(1)
        selected.append(pick)

    selected = pd.concat(selected).sort_values("score", ascending=False)

    # =========================
    # ⑤ TOP3
    # =========================
    selected = selected.head(TOP_N)

    # =========================
    # ⑥ EXIT（翌日寄り）
    # =========================
    new_positions = []

    for pos in positions:
        if i >= pos["exit_idx"]:
            exit_price = df[
                (df["Date"] == next_day) &
                (df["Ticker"] == pos["Ticker"])
            ]["Open"]

            if len(exit_price) > 0:
                ret = exit_price.values[0] / pos["entry_price"]
                capital *= ret
        else:
            new_positions.append(pos)

    positions = new_positions

    # =========================
    # ⑦ ENTRY（翌日寄り）
    # =========================
    slots = MAX_POSITIONS - len(positions)

    if slots > 0:
        entries = selected.head(slots)

        for _, row in entries.iterrows():

            entry_price = df[
                (df["Date"] == next_day) &
                (df["Ticker"] == row["Ticker"])
            ]["Open"]

            if len(entry_price) == 0:
                continue

            positions.append({
                "Ticker": row["Ticker"],
                "entry_price": entry_price.values[0],
                "exit_idx": i + HOLD_DAYS
            })

    # =========================
    # 記録
    # =========================
    equity_curve.append(capital)

# =========================
# 結果
# =========================
equity_curve = pd.Series(equity_curve)

returns = equity_curve.pct_change().fillna(0)

CAGR = (equity_curve.iloc[-1]) ** (252 / len(equity_curve)) - 1
Sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
MaxDD = (equity_curve / equity_curve.cummax() - 1).min()

print("\n=== RESULT ===")
print(f"CAGR  : {CAGR:.4f}")
print(f"Sharpe: {Sharpe:.4f}")
print(f"MaxDD : {MaxDD:.4f}")
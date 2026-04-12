import pandas as pd
import numpy as np
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
FEE = 0.001

# =========================
# データ読み込み
# =========================
df = pd.read_parquet(DATA_PATH)
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

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

# =========================
# Ranker用Target
# =========================
df["TargetRank"] = df.groupby("Date")["Target"].transform(
    lambda x: pd.qcut(x, q=N_CLASS, labels=False, duplicates="drop")
)

df = df.dropna(subset=["TargetRank"])
df["TargetRank"] = df["TargetRank"].astype(int)

# =========================
# モデル学習
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

model.fit(train_df[FEATURES], train_df["TargetRank"], group=group)

# =========================
# 🔥 高速化①：一括predict
# =========================
df["score"] = model.predict(df[FEATURES])

# =========================
# 🔥 高速化②：Date辞書
# =========================
date_groups = dict(tuple(df.groupby("Date")))
dates = sorted(date_groups.keys())

# =========================
# 🔥 高速化③：価格辞書
# =========================
price_dict = {
    (row.Date, row.Ticker): row.Open
    for row in df.itertuples()
}

# =========================
# バックテスト
# =========================
capital = INITIAL_CAPITAL
equity_curve = []
positions = []

trade_count = 0  # 🔥 追加

for i in range(len(dates) - HOLD_DAYS - 2):

    today = dates[i]
    next_day = dates[i + 1]

    today_df = date_groups[today]

    # =========================
    # ① FILTER
    # =========================
    today_df = today_df[today_df["TrendVol"] > -1.0]

    if len(today_df) == 0:
        equity_curve.append(capital)
        continue

    # =========================
    # ② TOP候補
    # =========================
    candidates = today_df.sort_values("score", ascending=False).head(CANDIDATE_N)

    # =========================
    # ③ DIVERSITY
    # =========================
    candidates = candidates.copy()

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
    # ④ TOP3
    # =========================
    selected = selected.head(TOP_N)

    # =========================
    # ⑤ EXIT
    # =========================
    realized_returns = []
    new_positions = []

    for pos in positions:
        if i >= pos["exit_idx"]:

            exit_price = price_dict.get((next_day, pos["Ticker"]))

            if exit_price is not None:
                ret = exit_price / pos["entry_price"] - 1
                ret -= FEE
                realized_returns.append(ret)

                trade_count += 1  # 🔥 カウント

        else:
            new_positions.append(pos)

    positions = new_positions

    # ポートフォリオ反映
    if len(realized_returns) > 0:
        capital *= (1 + np.mean(realized_returns))

    # =========================
    # ⑥ ENTRY
    # =========================
    slots = MAX_POSITIONS - len(positions)

    if slots > 0:
        entries = selected.head(slots)

        for _, row in entries.iterrows():

            entry_price = price_dict.get((next_day, row["Ticker"]))

            if entry_price is None:
                continue

            entry_price *= (1 + FEE)

            positions.append({
                "Ticker": row["Ticker"],
                "entry_price": entry_price,
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

print("\n=== RESULT（高速版） ===")
print(f"CAGR  : {CAGR:.4f}")
print(f"Sharpe: {Sharpe:.4f}")
print(f"MaxDD : {MaxDD:.4f}")
print(f"Trades: {trade_count}")
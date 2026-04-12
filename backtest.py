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

INITIAL_CAPITAL = 1.0
FEE = 0.001

DATA_PATH = "ml_dataset.parquet"

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year

FEATURES = [
    "Return_1","Return_3","MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility","Volume_change","Volume_ratio","HL_range",
    "Rel_Return_1","Trend_5_z","Trend_10_z","Trend_diff",
    "Gap","Volatility_change","Momentum_acc",
    "DD_5","DD_10","TrendVol","Volume_Z",
    "Return_1_rank","Volume_ratio_rank",
    "Trend_5_z_rank","TrendVol_rank",
    "Market_Z","Market_Trend"
]

df = df.dropna(subset=FEATURES + ["Target"]).copy()

# =========================
# Ranker target
# =========================
df["TargetRank"] = df.groupby("Date")["Target"].transform(
    lambda x: pd.qcut(x, q=N_CLASS, labels=False, duplicates="drop")
)
df = df.dropna(subset=["TargetRank"])
df["TargetRank"] = df["TargetRank"].astype(int)

# =========================
# 共通準備
# =========================
price_open = {
    (row.Date, row.Ticker): row.Open
    for row in df.itertuples()
}

all_results = []

# =========================
# 🔥 ウォークフォワード
# =========================
splits = [
    (2018, 2020, 2021),
    (2019, 2021, 2022),
    (2020, 2022, 2023),
]

for train_start, train_end, test_year in splits:

    print(f"\n=== WF: {train_start}-{train_end} → {test_year} ===")

    train_df = df[(df["Year"] >= train_start) & (df["Year"] <= train_end)].copy()
    test_df  = df[df["Year"] == test_year].copy()

    # =========================
    # モデル
    # =========================
    group = train_df.groupby("Date").size().tolist()

    model = LGBMRanker(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(train_df[FEATURES], train_df["TargetRank"], group=group)

    # =========================
    # スコア
    # =========================
    test_df["score"] = model.predict(test_df[FEATURES])

    dates = sorted(test_df["Date"].unique())
    date_groups = dict(tuple(test_df.groupby("Date")))

    # =========================
    # バックテスト
    # =========================
    capital = INITIAL_CAPITAL
    equity_curve = []
    positions = []

    for i in range(len(dates) - HOLD_DAYS - 1):

        today = dates[i]
        next_day = dates[i + 1]

        today_df = date_groups[today].copy()

        # =========================
        # 🔥 フィルタ無し
        # =========================
        today_df["final_score"] = today_df["score"]

        # =========================
        # EXIT
        # =========================
        daily_return = 0
        new_positions = []

        for pos in positions:

            if i == pos["exit_idx"]:

                exit_price = price_open.get((today, pos["Ticker"]))
                if exit_price is None:
                    continue

                ret = (exit_price / pos["entry_price"] - 1) - FEE
                daily_return += ret * pos["weight"]

            else:
                new_positions.append(pos)

        positions = new_positions

        # =========================
        # ENTRY
        # =========================
        candidates = today_df.sort_values("final_score", ascending=False).head(CANDIDATE_N)

        # diversity
        candidates["bucket"] = pd.qcut(
            candidates["TrendVol"],
            q=min(DIVERSITY_BUCKETS, len(candidates)),
            labels=False,
            duplicates="drop"
        )

        selected = []

        for b in sorted(candidates["bucket"].dropna().unique()):
            pick = candidates[candidates["bucket"] == b].head(1)
            selected.append(pick)

        selected = pd.concat(selected).sort_values("final_score", ascending=False).head(TOP_N)

        slots = MAX_POSITIONS - len(positions)

        if slots > 0:

            entries = selected.head(slots)

            weights = np.exp(entries["final_score"])
            weights /= weights.sum()

            for (_, row), w in zip(entries.iterrows(), weights):

                if any(p["Ticker"] == row["Ticker"] for p in positions):
                    continue

                entry_price = price_open.get((next_day, row["Ticker"]))
                if entry_price is None:
                    continue

                entry_price *= (1 + FEE)

                positions.append({
                    "Ticker": row["Ticker"],
                    "entry_price": entry_price,
                    "exit_idx": i + HOLD_DAYS,
                    "weight": w
                })

        # weight正規化
        if len(positions) > 0:
            total_w = sum(p["weight"] for p in positions)
            for p in positions:
                p["weight"] /= total_w

        capital *= (1 + daily_return)
        equity_curve.append(capital)

    # =========================
    # 評価
    # =========================
    equity_curve = pd.Series(equity_curve)
    returns = equity_curve.pct_change().fillna(0)

    CAGR = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
    Sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
    MaxDD = (equity_curve / equity_curve.cummax() - 1).min()

    print(f"CAGR  : {CAGR:.4f}")
    print(f"Sharpe: {Sharpe:.4f}")
    print(f"MaxDD : {MaxDD:.4f}")

    all_results.append({
        "Period": f"{train_start}-{train_end}→{test_year}",
        "CAGR": CAGR,
        "Sharpe": Sharpe,
        "MaxDD": MaxDD
    })

# =========================
# 全体まとめ
# =========================
print("\n=== WALK FORWARD SUMMARY ===")
print(pd.DataFrame(all_results))
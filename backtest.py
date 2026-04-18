import pandas as pd
import numpy as np
from lightgbm import LGBMRanker

# =========================
# 設定
# =========================
TOP_N = 3
MAX_POSITIONS = 5
HOLD_DAYS = 15
N_CLASS = 30

INITIAL_CAPITAL = 1.0
FEE = 0.001

DATA_PATH = "ml_dataset_15d.parquet"

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year

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

df = df.dropna(subset=FEATURES + ["Target"]).copy()

# =========================
# Rank用Target
# =========================
df["TargetRank"] = df.groupby("Date")["Target"].transform(
    lambda x: pd.qcut(x, q=N_CLASS, labels=False, duplicates="drop")
)

df = df.dropna(subset=["TargetRank"])
df["TargetRank"] = df["TargetRank"].astype(int)

# 価格辞書
price_open = {(r.Date, r.Ticker): r.Open for r in df.itertuples()}

# =========================
# 分割
# =========================
splits = [
    (2018, 2020, 2021),
    (2019, 2021, 2022),
    (2020, 2022, 2023),
]

results = []

# =========================
# 学習・検証
# =========================
for train_start, train_end, test_year in splits:

    print(f"\n=== {train_start}-{train_end} → {test_year} ===")

    train_df = df[(df["Year"] >= train_start) & (df["Year"] <= train_end)]
    test_df  = df[df["Year"] == test_year].copy()

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

    test_df["score_raw"] = model.predict(test_df[FEATURES])

    dates = sorted(test_df["Date"].unique())
    date_groups = dict(tuple(test_df.groupby("Date")))

    # 🔥 ここに追加
    print("dates:", dates[:5], "...", dates[-5:], flush=True)
    print("len(dates):", len(dates), flush=True)

    capital = INITIAL_CAPITAL
    positions = []
    equity = []

    for i in range(len(dates) - HOLD_DAYS - 1):

        today = dates[i]
        next_day = dates[i + 1]
        df_today = date_groups[today].copy()

        # =========================
        # 🔥 スコア正規化のみ
        # =========================
        sr = df_today["score_raw"]
        df_today["score_raw"] = (sr - sr.mean()) / (sr.std() + 1e-9)

        df_today["final_score"] = df_today["score_raw"]

        # =========================
        # EXIT
        # =========================
        daily_return = 0
        new_pos = []

        for p in positions:
            if i == p["exit_idx"]:
                price = price_open.get((today, p["Ticker"]))
                if price is not None:
                    ret = (price / p["entry"] - 1) - FEE
                    daily_return += ret * p["w"]
            else:
                new_pos.append(p)

        positions = new_pos

        # =========================
        # ENTRY（完全モデル依存）
        # =========================
        candidates = df_today.sort_values("final_score", ascending=False).head(TOP_N)

        slots = MAX_POSITIONS - len(positions)

        if slots > 0 and len(candidates) > 0:

            entries = candidates.head(slots)

            # 🔥 均等ウェイト
            weights = np.ones(len(entries)) / len(entries)

            for (_, r), w in zip(entries.iterrows(), weights):
                price = price_open.get((next_day, r["Ticker"]))
                if price is not None:
                    positions.append({
                        "Ticker": r["Ticker"],
                        "entry": price * (1 + FEE),
                        "exit_idx": i + HOLD_DAYS,
                        "w": w
                    })

        # =========================
        # weight正規化
        # =========================
        if positions:
            s = sum(p["w"] for p in positions)
            for p in positions:
                p["w"] /= s

        # =========================
        # 資産更新
        # =========================
        capital *= (1 + daily_return)
        equity.append(capital)

    equity = pd.Series(equity)
    ret = equity.pct_change().fillna(0)

    CAGR = equity.iloc[-1] ** (252 / len(equity)) - 1
    Sharpe = ret.mean() / (ret.std() + 1e-9) * np.sqrt(252)
    MaxDD = (equity / equity.cummax() - 1).min()

    print(f"CAGR  : {CAGR:.4f}")
    print(f"Sharpe: {Sharpe:.4f}")
    print(f"MaxDD : {MaxDD:.4f}")

    results.append({
        "Period": f"{train_start}-{train_end}→{test_year}",
        "CAGR": CAGR,
        "Sharpe": Sharpe,
        "MaxDD": MaxDD
    })

print("\n=== SUMMARY ===")
print(pd.DataFrame(results))
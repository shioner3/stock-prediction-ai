import pandas as pd
import numpy as np

# =========================
# 設定
# =========================
TOP_N_BASE = 3
MAX_POSITIONS = 5
HOLD_DAYS = 7

INITIAL_CAPITAL = 1.0
FEE = 0.001

DATA_PATH = "ml_dataset_7d.parquet"

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year

# =========================
# 🔥 スコア構築
# =========================
df["Score_TrendMomentum"] = df["Trend_20_z"] * df["Momentum_20"]

df["Score_Quality"] = (
    df["TrendVol_rank"]
    + (-df["DD_20_rank"])
)

df["Score_Reversal"] = -df["Return_5"]

# =========================
# 🔥 正規化（クリップ）
# =========================
for col in [
    "Score_TrendMomentum",
    "Score_Quality",
    "Score_Reversal"
]:
    df[col] = df.groupby("Date")[col].transform(
        lambda x: np.clip(
            (x - x.mean()) / (x.std() + 1e-9),
            -3, 3
        )
    )

# =========================
# 価格辞書
# =========================
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
# バックテスト
# =========================
for train_start, train_end, test_year in splits:

    print(f"\n=== {train_start}-{train_end} → {test_year} ===")

    test_df = df[df["Year"] == test_year].copy()

    dates = sorted(test_df["Date"].unique())
    date_groups = dict(tuple(test_df.groupby("Date")))

    capital = INITIAL_CAPITAL
    positions = []
    equity = []

    for i in range(len(dates) - HOLD_DAYS - 1):

        today = dates[i]
        next_day = dates[i + 1]
        df_today = date_groups[today].copy()

        # =========================
        # 🔥 市場フィルター（緩和）
        # =========================
        market_trend = df_today["Market_Trend"].iloc[0]

        if market_trend < -0.02:
            continue

        # =========================
        # 🔥 レジーム別調整
        # =========================
        if market_trend > 0.02:
            TOP_N = 5
            w_trend = 0.5
            w_rev = 0.05
        else:
            TOP_N = TOP_N_BASE
            w_trend = 0.4
            w_rev = 0.15

        # =========================
        # 🔥 スコア（シンプル化）
        # =========================
        df_today["final_score"] = (
            w_trend * df_today["Score_TrendMomentum"]
            + 0.35 * df_today["Score_Quality"]
            + w_rev * df_today["Score_Reversal"]
        )

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
        # ENTRY
        # =========================
        candidates = df_today.sort_values("final_score", ascending=False).head(10)

        if len(candidates) == 0:
            continue

        # 分散
        candidates["bucket"] = pd.qcut(
            candidates["TrendVol"],
            q=min(3, len(candidates)),
            labels=False,
            duplicates="drop"
        )

        selected = []

        for b in sorted(candidates["bucket"].dropna().unique()):
            tmp = candidates[candidates["bucket"] == b]
            if len(tmp) > 0:
                selected.append(tmp.head(1))

        if len(selected) > 0:
            selected = pd.concat(selected)
        else:
            selected = candidates.head(TOP_N)

        if len(selected) < TOP_N:
            remain = candidates[~candidates.index.isin(selected.index)]
            selected = pd.concat([selected, remain.head(TOP_N - len(selected))])

        slots = MAX_POSITIONS - len(positions)

        if slots > 0:
            entries = selected.head(slots)

            # 🔥 強いweight
            weights = np.exp(entries["final_score"] * 2.0)
            weights /= weights.sum()

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
        # 正規化
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

    if len(equity) == 0:
        print("⚠️ トレードなし")
        continue

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
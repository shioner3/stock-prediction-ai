import pandas as pd
import numpy as np

# =========================
# 設定（スコア戦略）
# =========================
TOP_N = 3
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
# 🔥 スコア構築（ここが全て）
# =========================

# トレンド × モメンタム
df["Score_TrendMomentum"] = df["Trend_20_z"] * df["Momentum_20"]

# 質（トレンド効率＋低DD）
df["Score_Quality"] = (
    df["TrendVol_rank"]
    + (-df["DD_20_rank"])
)

# 短期
df["Score_ShortTerm"] = (
    0.5 * df["Return_5"]
    + 0.5 * df["Return_10"]
)

# 逆張り
df["Score_Reversal"] = -df["Return_5"]

# =========================
# 🔥 正規化
# =========================
for col in [
    "Score_TrendMomentum",
    "Score_Quality",
    "Score_ShortTerm",
    "Score_Reversal"
]:
    df[col] = df.groupby("Date")[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )

# =========================
# 🔥 最終スコア
# =========================
df["final_score"] = (
    0.35 * df["Score_TrendMomentum"]
    + 0.30 * df["Score_Quality"]
    + 0.20 * df["Score_ShortTerm"]
    + 0.15 * df["Score_Reversal"]
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
        # 🔥 市場レジーム
        # =========================
        market_trend = df_today["Market_Trend"].iloc[0]
        market_sharpe = df_today["Market_Sharpe"].iloc[0]

        if market_trend < 0:
            w_trend = 0.2
            w_rev = 0.3
        else:
            w_trend = 0.4
            w_rev = 0.1

        df_today["final_score"] = (
            w_trend * df_today["Score_TrendMomentum"]
            + 0.30 * df_today["Score_Quality"]
            + 0.20 * df_today["Score_ShortTerm"]
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
        # ENTRY（スコアのみ）
        # =========================
        candidates = df_today.sort_values("final_score", ascending=False).head(TOP_N)

        slots = MAX_POSITIONS - len(positions)

        if slots > 0 and len(candidates) > 0:

            entries = candidates.head(slots)

            # 🔥 スコア比例ウェイト
            weights = np.exp(entries["final_score"])
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
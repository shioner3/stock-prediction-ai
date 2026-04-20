import pandas as pd
import numpy as np
from lightgbm import LGBMRanker

# =========================
# 設定（AI主体・4日戦略）
# =========================
DATA_PATH = "ml_dataset_4d.parquet"

HOLD_DAYS = 4
TOP_N_BASE = 3
TOP_RATE_BASE = 0.02

INITIAL_CAPITAL = 1.0
FEE = 0.001

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

# =========================
# 市場トレンド
# =========================
market = df.groupby("Date")["Close"].mean()
market = market.pct_change(5)
df = df.merge(market.rename("Market_Trend"), on="Date")

# =========================
# 特徴量
# =========================
FEATURES = [
    "Breakout",
    "Volume_Spike",
    "Vol_Expansion",
    "Gap",
    "Trend_5",
    "Trend_10",
    "Momentum_Std",
    "Drawdown",
    "final_score"
]

df = df.dropna(subset=FEATURES + ["TargetRank", "Market_Trend"]).copy()

# =========================
# 価格辞書
# =========================
price_open = {(r.Date, r.Ticker): r.Open for r in df.itertuples()}

# =========================
# 期間分割
# =========================
splits = [
    (2018, 2021, 2022),
    (2019, 2022, 2023),
    (2020, 2023, 2024),
]

results = []

# =========================
# ループ
# =========================
for train_start, train_end, test_year in splits:

    print(f"\n=== {train_start}-{train_end} → {test_year} ===")

    train_df = df[(df["Year"] >= train_start) & (df["Year"] <= train_end)].copy()
    test_df = df[df["Year"] == test_year].copy()

    # =========================
    # ラベル
    # =========================
    train_df["TargetRankInt"] = pd.qcut(
        train_df["TargetRank"],
        q=10,
        labels=False,
        duplicates="drop"
    ).astype(int)

    # =========================
    # 学習
    # =========================
    X = train_df[FEATURES]
    y = train_df["TargetRankInt"]
    group = train_df.groupby("Date").size().values

    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X, y, group=group)

    # =========================
    # 予測
    # =========================
    test_df["pred_score"] = model.predict(test_df[FEATURES])

    dates = sorted(test_df["Date"].unique())
    date_groups = dict(tuple(test_df.groupby("Date")))

    capital = INITIAL_CAPITAL
    positions = []
    equity = []

    # 🔥 トレード数カウンター
    trade_count = 0

    # =========================
    # バックテスト
    # =========================
    for i in range(len(dates) - HOLD_DAYS - 1):

        today = dates[i]
        next_day = dates[i + 1]
        df_today = date_groups[today].copy()

        # =========================
        # レジーム
        # =========================
        market_trend = df_today["Market_Trend"].iloc[0]

        if market_trend < 0.01:
            equity.append(capital)
            continue

        if market_trend > 0.02:
            TOP_N = 5
            TOP_RATE = 0.03
        else:
            TOP_N = TOP_N_BASE
            TOP_RATE = TOP_RATE_BASE

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
        df_today["pred_rank"] = df_today["pred_score"].rank(ascending=False, pct=True)

        candidates = df_today[
            (df_today["pred_rank"] <= TOP_RATE) &
            (df_today["pred_score"] > 0) &
            (df_today["Trend_5"] > 0) &
            (df_today["Volume_Spike"] > 0) &
            (df_today["Return_3"] < 0.1) &
            (df_today["Drawdown"] > -0.05)
        ]

        if len(candidates) > 0:

            selected = candidates.sort_values("pred_score", ascending=False).head(TOP_N)

            slots = TOP_N - len(positions)

            if slots > 0:
                entries = selected.head(slots)

                weights = np.exp(entries["pred_score"] * 1.5)
                weights /= weights.sum()

                for (_, r), w in zip(entries.iterrows(), weights):
                    price = price_open.get((next_day, r["Ticker"]))
                    if price is not None:
                        trade_count += 1  # 🔥ここ追加

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

    # =========================
    # 評価
    # =========================
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
    print(f"Trades: {trade_count}")  # 🔥表示

    results.append({
        "Period": f"{train_start}-{train_end}→{test_year}",
        "CAGR": CAGR,
        "Sharpe": Sharpe,
        "MaxDD": MaxDD,
        "Trades": trade_count
    })

print("\n=== SUMMARY ===")
print(pd.DataFrame(results))
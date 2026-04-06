import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

INITIAL_CAPITAL = 1.0
MAX_POSITIONS = 10
HOLD_DAYS = 10

TREND_TH = 0.0
N_BINS = 10          # 🔥 スコア分割数
MIN_RET = 0.0        # 🔥 採用する最低期待値

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

FEATURES = [
    "Return_1","Return_3",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio",
    "HL_range",
    "Rel_Return_1",
    "Trend_5_z","Trend_10_z",
    "Gap","Volatility_change","Volume_spike","Momentum_acc"
]

# =========================
# レジーム
# =========================
market = df.groupby("Date")["Return_1"].mean()
market_smooth = market.rolling(20).mean()

df["Market_Smooth"] = df["Date"].map(market_smooth)

df["Regime"] = np.where(
    df["Market_Smooth"] > 0.001, "up",
    np.where(df["Market_Smooth"] < -0.001, "down", "range")
)

# =========================
# モデル
# =========================
def train_model(train_df):
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(train_df[FEATURES], train_df["Target"])
    return model

# =========================
# 🔥 スコア帯最適化
# =========================
def optimize_score_range(train_df, model):

    df_tmp = train_df.copy()
    df_tmp["raw_score"] = model.predict(df_tmp[FEATURES])

    # 🔥 境界取得
    _, bins = pd.qcut(df_tmp["raw_score"], N_BINS, retbins=True, duplicates="drop")

    df_tmp["bin"] = pd.cut(df_tmp["raw_score"], bins=bins, include_lowest=True)

    stats = df_tmp.groupby("bin")["Target"].mean()

    good_bins = stats[stats > MIN_RET].index

    print("\n=== SCORE BIN OPT ===")
    print(stats)
    print("USE BINS:", list(good_bins))

    return bins, good_bins
# =========================
# バックテスト
# =========================
def run_backtest(train_df, test_df):

    model = train_model(train_df)

    # 🔥 スコア帯を学習
    good_bins = optimize_score_range(train_df, model)

    test_df = test_df.copy()
    test_df["raw_score"] = model.predict(test_df[FEATURES])
    test_df["score"] = test_df.groupby("Date")["raw_score"].rank(pct=True)

    # testにもbin適用
    test_df["bin"] = pd.qcut(test_df["raw_score"], N_BINS, duplicates="drop")

    dates = sorted(test_df["Date"].unique())
    grouped = {d: g for d, g in test_df.groupby("Date")}

    equity = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    equity_curve = []

    positions = []
    trade_logs = []

    for i, d in enumerate(dates):

        today = grouped[d]

        # =========================
        # 決済
        # =========================
        new_positions = []

        for pos in positions:
            if i == pos["exit_idx"]:

                cur = today[today["Ticker"] == pos["ticker"]]
                if cur.empty:
                    continue

                exit_price = cur["Open"].iloc[0]
                ret = (exit_price - pos["entry_price"]) / pos["entry_price"]

                cash += pos["capital"] * (1 + ret)

                trade_logs.append({
                    "ticker": pos["ticker"],
                    "entry_date": pos["entry_date"],
                    "exit_date": d,
                    "return": ret,
                    "raw_score": pos["raw_score"],
                    "score": pos["score"],
                    "regime": pos["regime"]
                })

            else:
                new_positions.append(pos)

        positions = new_positions

        # =========================
        # エントリー（🔥down特化 + スコア帯）
        # =========================
        if i + 1 < len(dates):

            available = MAX_POSITIONS - len(positions)

            if available > 0 and cash > 0:

                next_day = dates[i + 1]
                next_data = grouped[next_day]

                today_f = today[
                    (today["Trend_5_z"] > TREND_TH) &
                    (today["Regime"] == "down") &
                    (today["bin"].isin(good_bins))  # 🔥ここが核心
                ]

                if len(today_f) > 0:

                    picks = today_f.nlargest(available, "score")

                    capital_per_position = cash / len(picks)

                    for row in picks.itertuples():

                        ticker = row.Ticker

                        if any(p["ticker"] == ticker for p in positions):
                            continue

                        next_row = next_data[next_data["Ticker"] == ticker]
                        if next_row.empty:
                            continue

                        exit_idx = i + HOLD_DAYS
                        if exit_idx >= len(dates):
                            continue

                        positions.append({
                            "ticker": ticker,
                            "entry_price": next_row["Open"].iloc[0],
                            "entry_date": d,
                            "exit_idx": exit_idx,
                            "capital": capital_per_position,
                            "raw_score": row.raw_score,
                            "score": row.score,
                            "regime": row.Regime
                        })

                        cash -= capital_per_position

        # =========================
        # 評価
        # =========================
        pos_val = 0

        for pos in positions:
            cur = today[today["Ticker"] == pos["ticker"]]
            if not cur.empty:
                price = cur["Close"].iloc[0]
                ret = (price - pos["entry_price"]) / pos["entry_price"]
                pos_val += pos["capital"] * (1 + ret)

        equity = cash + pos_val
        equity_curve.append(equity)

    # =========================
    # 評価指標
    # =========================
    equity_curve = pd.Series(equity_curve)
    returns = equity_curve.pct_change().dropna()

    CAGR = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
    Sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
    MaxDD = (equity_curve / equity_curve.cummax() - 1).min()

    trade_df = pd.DataFrame(trade_logs)

    return (CAGR, Sharpe, MaxDD), trade_df


# =========================
# 実行
# =========================
all_metrics = []

for y in sorted(df["Date"].dt.year.unique()):

    if y < 2022:
        continue

    train_df = df[df["Date"].dt.year < y]
    test_df = df[df["Date"].dt.year == y]

    res, _ = run_backtest(train_df, test_df)

    if res is not None:
        all_metrics.append(res)

# =========================
# 結果
# =========================
if len(all_metrics) > 0:

    cagr = np.mean([m[0] for m in all_metrics])
    sharpe = np.mean([m[1] for m in all_metrics])
    mdd = np.mean([m[2] for m in all_metrics])

    print("\n=== RESULT ===")
    print(f"CAGR  : {cagr:.4f}")
    print(f"Sharpe: {sharpe:.4f}")
    print(f"MaxDD : {mdd:.4f}")
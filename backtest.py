import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

INITIAL_CAPITAL = 1.0
MAX_POSITIONS = 10   # 🔥 分散強化（重要）
HOLD_DAYS = 10

TREND_TH = 0.0
SCORE_TH = 0.7       # 🔥 スコア下限
VOL_Q = 0.8          # 🔥 ボラ上限

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
# バックテスト
# =========================
def run_backtest(train_df, test_df):

    model = train_model(train_df)

    test_df = test_df.copy()
    test_df["raw_score"] = model.predict(test_df[FEATURES])
    test_df["score"] = test_df.groupby("Date")["raw_score"].rank(pct=True)

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
        # エントリー（🔥フィルタ強化）
        # =========================
        if i + 1 < len(dates):

            available = MAX_POSITIONS - len(positions)

            if available > 0 and cash > 0:

                next_day = dates[i + 1]
                next_data = grouped[next_day]

                # =========================
                # 🔥 フィルタ群
                # =========================

                # ① トレンド
                today_f = today[today["Trend_5_z"] > TREND_TH]

                # ② スコア下限
                today_f = today_f[today_f["score"] > SCORE_TH]

                # ③ ボラフィルタ（過熱除去）
                if len(today_f) > 0:
                    vol_th = today_f["Volatility"].quantile(VOL_Q)
                    today_f = today_f[today_f["Volatility"] < vol_th]

                # ④ 市場フィルタ（上昇相場のみ）
                today_f = today_f[today_f["Market_Smooth"] > 0]

                # ⑤ 出来高スパイク（ブレイクアウト狙い）
                today_f = today_f[today_f["Volume_spike"] > 1.0]

                if len(today_f) > 0:

                    # 上位選択
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

    # =========================
    # raw_score検証
    # =========================
    print("\n=== RAW SCORE CHECK ===")

    try:
        q10 = trade_df.groupby(
            pd.qcut(trade_df["raw_score"], 10, duplicates="drop")
        )["return"].mean()

        print(q10)

        corr = np.corrcoef(
            trade_df["raw_score"],
            trade_df["return"]
        )[0, 1]

        print("corr:", corr)

    except:
        pass

    return (CAGR, Sharpe, MaxDD), trade_df


# =========================
# 実行
# =========================
all_metrics = []
all_trades = []

for y in sorted(df["Date"].dt.year.unique()):

    if y < 2022:
        continue

    train_df = df[df["Date"].dt.year < y]
    test_df = df[df["Date"].dt.year == y]

    res, trades = run_backtest(train_df, test_df)

    if res is not None:
        all_metrics.append(res)

    if trades is not None:
        all_trades.append(trades)

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

# =========================
# ログ分析
# =========================
if len(all_trades) > 0:

    trade_log_df = pd.concat(all_trades, ignore_index=True)

    print("\n=== REGIME ANALYSIS ===")
    print(trade_log_df.groupby("regime")["return"].mean())

    print("\n=== WIN RATE ===")
    print((trade_log_df["return"] > 0).mean())
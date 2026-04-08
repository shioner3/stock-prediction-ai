import pandas as pd
import numpy as np
from lightgbm import LGBMRanker

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

INITIAL_CAPITAL = 1.0
MAX_POSITIONS = 5

TOP_N = 3
TOP_RATE = 0.005   # 🔥 変更
HOLD_DAYS = 7

USE_MARKET_FILTER = True
N_CLASS = 30

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
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
    "Gap",
    "Volatility_change",
    "Momentum_acc",
    "Return_1_rank","Return_3_rank",
    "Volume_ratio_rank","Trend_5_z_rank","HL_range_rank",
    "Market_Z","Market_Trend",
    "DayOfWeek"
]

# =========================
# TargetClass
# =========================
def make_target_class(x):
    try:
        return pd.qcut(x, q=N_CLASS, labels=False, duplicates="drop")
    except:
        return pd.cut(x, bins=min(N_CLASS, len(x)), labels=False)

df["TargetClass"] = df.groupby("Date")["Target"].transform(make_target_class).astype(int)

# =========================
# モデル（高速化版）
# =========================
def train_model(train_df):

    train_df = train_df.sort_values("Date")
    group = train_df.groupby("Date").size().to_list()

    model = LGBMRanker(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train_df[FEATURES],
        train_df["TargetClass"],
        group=group
    )

    return model

# =========================
# バックテスト（高速化）
# =========================
def run_backtest(train_df, test_df):

    model = train_model(train_df)

    test_df = test_df.copy()
    test_df["raw_score"] = model.predict(test_df[FEATURES])
    test_df["score"] = test_df.groupby("Date")["raw_score"].rank(pct=True)

    grouped = {d: g.set_index("Ticker") for d, g in test_df.groupby("Date")}
    dates = sorted(grouped.keys())

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

                if pos["ticker"] not in today.index:
                    continue

                exit_price = today.loc[pos["ticker"], "Open"]
                ret = (exit_price - pos["entry_price"]) / pos["entry_price"]

                cash += pos["capital"] * (1 + ret)
                trade_logs.append(ret)

            else:
                new_positions.append(pos)

        positions = new_positions

        # =========================
        # エントリー
        # =========================
        if i + 1 < len(dates):

            next_day = dates[i + 1]
            next_data = grouped[next_day]

            today_f = today.copy()

            # 🔥 市場フィルター（強化）
            if USE_MARKET_FILTER:
                today_f = today_f[today_f["Market_Trend"] > 0.003]

            # 🔥 トレンド強化
            today_f = today_f[today_f["Trend_5_z"] > 1.0]

            # 🔥 TOP_RATE（強化）
            today_f = today_f[today_f["score"] >= (1 - TOP_RATE)]

            if len(today_f) > 0:

                picks = today_f.sort_values("score", ascending=False).head(TOP_N)

                # 🔥 ウェイト（そのまま維持）
                weights = (picks["score"] ** 2) * (1 + picks["Trend_5_z"].clip(0, 2))

                if weights.sum() == 0:
                    continue

                weights = weights / weights.sum()

                for j, (ticker, row) in enumerate(picks.iterrows()):

                    if ticker not in next_data.index:
                        continue

                    capital = cash * weights.iloc[j]
                    capital = min(capital, equity * 0.2)

                    exit_idx = i + HOLD_DAYS
                    if exit_idx >= len(dates):
                        continue

                    positions.append({
                        "ticker": ticker,
                        "entry_price": next_data.loc[ticker, "Open"],
                        "exit_idx": exit_idx,
                        "capital": capital
                    })

                    cash -= capital

        # =========================
        # 評価
        # =========================
        pos_val = 0
        for pos in positions:
            if pos["ticker"] in today.index:
                price = today.loc[pos["ticker"], "Close"]
                ret = (price - pos["entry_price"]) / pos["entry_price"]
                pos_val += pos["capital"] * (1 + ret)

        equity = cash + pos_val
        equity_curve.append(equity)

    equity_curve = pd.Series(equity_curve)
    returns = equity_curve.pct_change().dropna()

    trade_df = pd.Series(trade_logs)

    if len(equity_curve) < 50 or trade_df.empty:
        return None

    CAGR = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
    Sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
    MaxDD = (equity_curve / equity_curve.cummax() - 1).min()

    win_rate = (trade_df > 0).mean()
    avg_win = trade_df[trade_df > 0].mean()
    avg_loss = trade_df[trade_df < 0].mean()

    pf = -avg_win / avg_loss if avg_loss != 0 else np.nan
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    trade_count = len(trade_df)

    return {
        "CAGR": CAGR,
        "Sharpe": Sharpe,
        "MaxDD": MaxDD,
        "WinRate": win_rate,
        "PF": pf,
        "Expectancy": expectancy,
        "Trades": trade_count
    }

# =========================
# 実行
# =========================
results = []

for y in sorted(df["Date"].dt.year.unique()):
    if y < 2022:
        continue

    train_df = df[df["Date"].dt.year < y]
    test_df = df[df["Date"].dt.year == y]

    res = run_backtest(train_df, test_df)

    if res:
        results.append(res)

# =========================
# 出力
# =========================
for i, r in enumerate(results):

    print(f"\n=== YEAR {i} ===")

    print("\n--- 基本 ---")
    print(f"CAGR  : {r['CAGR']:.3f}")
    print(f"Sharpe: {r['Sharpe']:.3f}")
    print(f"MaxDD : {r['MaxDD']:.3f}")

    print("\n--- 勝ちの質 ---")
    print(f"WinRate   : {r['WinRate']:.3f}")
    print(f"PF        : {r['PF']:.3f}")
    print(f"Expectancy: {r['Expectancy']:.4f}")

    print("\n--- 取引数 ---")
    print(f"Trades: {r['Trades']}")
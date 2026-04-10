import pandas as pd
import numpy as np
from lightgbm import LGBMRanker
import random

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

INITIAL_CAPITAL = 1.0

TOP_N = 5
HOLD_DAYS = 10

TREND_TH = 1.0
TOP_RATE = 0.015

USE_MARKET_FILTER = True
N_CLASS = 30

MAX_TICKERS = 300
STOP_LOSS = -0.02
COST_RATE = 0.0025
SLIPPAGE = 0.002

N_TRIALS = 10

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

top_tickers = df["Ticker"].value_counts().head(MAX_TICKERS).index
df = df[df["Ticker"].isin(top_tickers)]
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
    "DD_5","DD_10",
    "TrendVol",
    "Volume_Z",
    "Gap",
    "Volatility_change",
    "Momentum_acc",
    "Return_1_rank","Return_3_rank",
    "Volume_ratio_rank","Trend_5_z_rank","HL_range_rank",
    "Market_Z","Market_Trend",
    "DayOfWeek"
]

# =========================
# Target
# =========================
def make_target_class(x):
    try:
        return pd.qcut(x, q=N_CLASS, labels=False, duplicates="drop")
    except:
        return pd.cut(x, bins=min(N_CLASS, len(x)), labels=False)

df["TargetClass"] = df.groupby("Date")["Target"].transform(make_target_class).astype(int)

# =========================
# モデル
# =========================
def train_model(train_df):
    train_df = train_df.sort_values("Date")
    group = train_df.groupby("Date").size().to_list()

    model = LGBMRanker(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )

    model.fit(train_df[FEATURES], train_df["TargetClass"], group=group)
    return model

# =========================
# 可変ホールド
# =========================
def calc_hold_days(row):
    base = HOLD_DAYS
    trend_bonus = int(row["Trend_5_z"] * 2)
    stability_bonus = int((1 - row["TrendVol"]) * 5)
    dd_bonus = int((row["DD_5"] + 0.1) * 5)

    hold = base + trend_bonus + stability_bonus + dd_bonus
    return max(5, min(20, hold))

# =========================
# 最大連敗
# =========================
def calc_max_losing_streak(trades):
    streak = 0
    max_streak = 0
    for r in trades:
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak

# =========================
# バックテスト
# =========================
def run_backtest(model, data_df):

    data_df = data_df.copy().dropna(subset=FEATURES)

    data_df["raw_score"] = model.predict(data_df[FEATURES])
    data_df["score"] = data_df.groupby("Date")["raw_score"].rank(pct=True)

    grouped = {d: g.set_index("Ticker") for d, g in data_df.groupby("Date")}
    dates = sorted(grouped.keys())

    equity = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    positions = []

    trade_logs = []
    equity_curve = []

    for i, d in enumerate(dates):

        today = grouped[d]

        # ===== 決済 =====
        new_positions = []
        for pos in positions:

            if pos["ticker"] not in today.index:
                continue

            price = today.loc[pos["ticker"], "Close"]
            ret = (price - pos["entry_price"]) / pos["entry_price"]

            if ret <= STOP_LOSS or i == pos["exit_idx"]:

                exit_price = today.loc[pos["ticker"], "Open"] * (1 - SLIPPAGE)
                final_ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                final_ret -= COST_RATE * 2

                cash += pos["capital"] * (1 + final_ret)
                trade_logs.append(final_ret)

            else:
                new_positions.append(pos)

        positions = new_positions

        # ===== エントリー =====
        if i + 1 < len(dates):

            next_data = grouped[dates[i+1]]
            today_f = today.copy()

            if USE_MARKET_FILTER:
                today_f = today_f[today_f["Market_Trend"] > 0]

            today_f = today_f[today_f["Trend_5_z"] > TREND_TH]
            today_f = today_f[today_f["score"] >= (1 - TOP_RATE)]

            # スコア統合
            today_f["adjusted_score"] = (
                today_f["score"]
                * (1 + today_f["Trend_5_z"].clip(0, 2))
                * (1 - today_f["TrendVol"].clip(0, 1))
                * (1 + today_f["DD_5"].clip(-0.2, 0.2))
                * (1 + today_f["Market_Trend"].clip(0, 0.02))
            )

            if len(today_f) > 0:

                picks = today_f.sort_values("adjusted_score", ascending=False).head(TOP_N)

                weights = picks["adjusted_score"]

                if weights.sum() > 0:
                    weights = (weights / weights.sum()).values
                else:
                    weights = np.ones(len(weights)) / len(weights)

                for j, (ticker, row) in enumerate(picks.iterrows()):

                    if ticker not in next_data.index:
                        continue

                    capital = min(cash * weights[j], equity * 0.2)

                    hold_days = calc_hold_days(row)
                    exit_idx = i + hold_days

                    if exit_idx >= len(dates):
                        continue

                    entry_price = next_data.loc[ticker, "Open"] * (1 + SLIPPAGE + COST_RATE)

                    positions.append({
                        "ticker": ticker,
                        "entry_price": entry_price,
                        "exit_idx": exit_idx,
                        "capital": capital
                    })

                    cash -= capital

        # ===== エクイティ =====
        pos_val = 0
        for pos in positions:
            if pos["ticker"] in today.index:
                price = today.loc[pos["ticker"], "Close"]
                ret = (price - pos["entry_price"]) / pos["entry_price"]
                pos_val += pos["capital"] * (1 + ret)

        equity = cash + pos_val
        equity_curve.append(equity)

    if len(trade_logs) < 50:
        return None

    trade_df = pd.Series(trade_logs)
    equity_df = pd.DataFrame({"Equity": equity_curve})

    equity_df["Return"] = equity_df["Equity"].pct_change().fillna(0)

    # ===== MaxDD =====
    peak = equity_df["Equity"].cummax()
    dd = equity_df["Equity"] / peak - 1
    max_dd = dd.min()

    # ===== CAGR =====
    years = len(equity_df) / 252
    final_equity = equity_df["Equity"].iloc[-1]
    cagr = final_equity ** (1 / years) - 1 if years > 0 else 0

    # ===== Sharpe =====
    sharpe = (
        equity_df["Return"].mean() / equity_df["Return"].std()
    ) * np.sqrt(252) if equity_df["Return"].std() != 0 else 0

    # ===== 最大連敗 =====
    max_ls = calc_max_losing_streak(trade_logs)

    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "LosingStreak": max_ls,
        "Trades": len(trade_df)
    }

# =========================
# ランダム検証
# =========================
results = []

all_years = sorted(df["Date"].dt.year.unique())

for i in range(N_TRIALS):

    train_years = random.sample(list(all_years[:-1]), 3)
    test_year = random.choice([y for y in all_years if y not in train_years])

    train_df = df[df["Date"].dt.year.isin(train_years)]
    test_df  = df[df["Date"].dt.year == test_year]

    print(f"\nTrial {i+1}: Train {train_years} → Test {test_year}")

    model = train_model(train_df)
    res = run_backtest(model, test_df)

    if res is None:
        continue

    res["Train"] = train_years
    res["Test"] = test_year

    results.append(res)

# =========================
# 結果
# =========================
result_df = pd.DataFrame(results)

print("\n=== ランダム検証結果 ===")
print(result_df)

print("\n平均CAGR:", result_df["CAGR"].mean())
print("平均Sharpe:", result_df["Sharpe"].mean())
print("平均MaxDD:", result_df["MaxDD"].mean())
print("平均連敗数:", result_df["LosingStreak"].mean())
print("平均Trades:", result_df["Trades"].mean())
import pandas as pd
import numpy as np
from lightgbm import LGBMRanker

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

INITIAL_CAPITAL = 1.0
TOP_N = 5
HOLD_DAYS = 5

STOP_LOSS = -0.03
COST_RATE = 0.0025
SLIPPAGE = 0.002

MAX_POSITION_RATIO = 0.2  # 1銘柄最大20%

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

# =========================
# Target（リーク対策）
# =========================
df["Target"] = (
    df.groupby("Ticker")["Close"].shift(-HOLD_DAYS) / df["Close"] - 1
).clip(-0.2, 0.2)

# Target生成後に削除（ここ重要）
df = df.dropna(subset=["Target"])

# =========================
# Features
# =========================
FEATURES = [
    "Return_1","Return_3",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility","Volume_change","Volume_ratio",
    "HL_range","Rel_Return_1",
    "Trend_5_z","Trend_10_z","Trend_diff",
    "DD_5","DD_10",
    "TrendVol","Volume_Z",
    "Gap","Volatility_change","Momentum_acc",
    "Return_1_rank","Return_3_rank",
    "Volume_ratio_rank","Trend_5_z_rank","HL_range_rank",
    "Market_Z","Market_Trend",
    "DayOfWeek"
]

df = df.dropna(subset=FEATURES)

# =========================
# TargetClass
# =========================
def make_target_class(x):
    if len(x) < 30:
        return pd.Series([np.nan]*len(x), index=x.index)
    try:
        return pd.qcut(x, q=30, labels=False, duplicates="drop")
    except:
        return pd.Series([np.nan]*len(x), index=x.index)

df["TargetClass"] = df.groupby("Date")["Target"].transform(make_target_class)
df = df.dropna(subset=["TargetClass"])
df["TargetClass"] = df["TargetClass"].astype(int)

# =========================
# モデル
# =========================
def train_model(train_df):

    train_df = train_df.sort_values("Date")
    group = train_df.groupby("Date").size().to_list()

    model = LGBMRanker(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(train_df[FEATURES], train_df["TargetClass"], group=group)
    return model

# =========================
# バックテスト
# =========================
def run_backtest(model, data_df):

    data_df = data_df.copy()

    # スコア
    data_df["raw_score"] = model.predict(data_df[FEATURES])
    data_df["score"] = data_df.groupby("Date")["raw_score"].rank(pct=True)
    data_df["score_shift"] = data_df.groupby("Ticker")["score"].shift(1)

    grouped = {d: g.set_index("Ticker") for d, g in data_df.groupby("Date")}
    dates = sorted(grouped.keys())

    equity = cash = INITIAL_CAPITAL
    positions = []

    equity_curve = []
    trade_returns = []

    for i, d in enumerate(dates):

        today = grouped[d]

        # =========================
        # 決済
        # =========================
        new_positions = []

        for pos in positions:

            if pos["ticker"] not in today.index:
                continue

            open_p = today.loc[pos["ticker"], "Open"]
            low_p  = today.loc[pos["ticker"], "Low"]

            stop_price = pos["entry_price"] * (1 + STOP_LOSS)

            if low_p <= stop_price or i >= pos["exit_idx"]:

                exit_price = min(open_p, stop_price) * (1 - SLIPPAGE)
                ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                ret -= COST_RATE * 2

                cash += pos["capital"] * (1 + ret)
                trade_returns.append(ret)

            else:
                new_positions.append(pos)

        positions = new_positions

        # =========================
        # エントリー
        # =========================
        if i + 1 < len(dates):

            today_f = today.copy()
            today_f = today_f.dropna(subset=["score_shift"])

            # フィルタ
            today_f = today_f[
                (today_f["Market_Trend"] > 0) &
                (today_f["Trend_5_z"] > 1.0) &
                (today_f["score_shift"] > 0.97)
            ]

            # ノートレ
            if len(today_f) > 0:

                picks = today_f.sort_values("score_shift", ascending=False).head(TOP_N)

                weights = np.exp(picks["score_shift"] * 5)
                weights = weights / weights.sum()

                next_data = grouped[dates[i+1]]

                for j, (ticker, row) in enumerate(picks.iterrows()):

                    if ticker not in next_data.index:
                        continue

                    capital = min(cash * weights.iloc[j], equity * MAX_POSITION_RATIO)

                    if capital <= 0:
                        continue

                    exit_idx = i + HOLD_DAYS
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

    # =========================
    # 指標
    # =========================
    equity_series = pd.Series(equity_curve)

    returns = equity_series.pct_change().fillna(0)

    years = len(equity_series) / 252
    cagr = equity_series.iloc[-1] ** (1 / years) - 1

    sharpe = (
        returns.mean() / returns.std() * np.sqrt(252)
        if returns.std() != 0 else 0
    )

    peak = equity_series.cummax()
    max_dd = (equity_series / peak - 1).min()

    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "Trades": len(trade_returns)
    }

# =========================
# 実行
# =========================
years = sorted(df["Date"].dt.year.unique())

results = []

for i in range(3, len(years)):

    train = df[df["Date"].dt.year < years[i]]
    test  = df[df["Date"].dt.year == years[i]]

    model = train_model(train)
    res = run_backtest(model, test)

    res["Year"] = years[i]
    results.append(res)

result_df = pd.DataFrame(results)

print("\n=== Yearly Performance ===")
print(result_df[["Year","CAGR","Sharpe","MaxDD","Trades"]])

print("\n=== Summary ===")
print(f"CAGR   : {result_df['CAGR'].mean():.3f}")
print(f"Sharpe : {result_df['Sharpe'].mean():.3f}")
print(f"MaxDD  : {result_df['MaxDD'].mean():.3f}")
print(f"Trades : {result_df['Trades'].mean():.1f}")
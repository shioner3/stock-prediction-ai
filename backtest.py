import pandas as pd
import numpy as np
from lightgbm import LGBMRanker

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

INITIAL_CAPITAL = 1.0

TOP_N = 5
TOP_RATE = 0.005
HOLD_DAYS = 10

USE_MARKET_FILTER = True
N_CLASS = 30

MAX_TICKERS = 1000

# 🔥 固定損切り（ここが最終形）
STOP_LOSS = -0.02

# 🔥 コスト（超重要）
COST_RATE = 0.0015  # 0.15%片道想定

WF_PERIODS = [
    ([2018, 2019, 2020], 2021),
    ([2019, 2020, 2021], 2022),
    ([2020, 2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]

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

    model.fit(
        train_df[FEATURES],
        train_df["TargetClass"],
        group=group
    )

    return model

# =========================
# バックテスト（完全版）
# =========================
def run_backtest(model, data_df):

    data_df = data_df.copy()
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

        # =========================
        # 決済
        # =========================
        new_positions = []

        for pos in positions:

            if pos["ticker"] not in today.index:
                continue

            price = today.loc[pos["ticker"], "Close"]
            ret = (price - pos["entry_price"]) / pos["entry_price"]

            # 🔥 損切り or 時間決済
            if ret <= STOP_LOSS or i == pos["exit_idx"]:

                exit_price = today.loc[pos["ticker"], "Open"]

                final_ret = (exit_price - pos["entry_price"]) / pos["entry_price"]

                # 🔥 コスト反映（売買両方）
                final_ret -= COST_RATE * 2

                cash += pos["capital"] * (1 + final_ret)
                trade_logs.append(final_ret)

            else:
                new_positions.append(pos)

        positions = new_positions

        # =========================
        # エントリー
        # =========================
        if i + 1 < len(dates):

            if today["Market_Trend"].mean() < 0:
                equity_curve.append(equity)
                continue

            next_data = grouped[dates[i+1]]
            today_f = today.copy()

            if USE_MARKET_FILTER:
                today_f = today_f[today_f["Market_Trend"] > 0.008]

            today_f = today_f[today_f["Trend_5_z"] > 1.2]
            today_f = today_f[today_f["score"] >= (1 - TOP_RATE)]

            if len(today_f) > 0:

                picks = today_f.sort_values("score", ascending=False).head(TOP_N)

                weights = (picks["score"] ** 2) * (1 + picks["Trend_5_z"].clip(0, 2))
                weights = weights / weights.sum()

                for j, (ticker, row) in enumerate(picks.iterrows()):

                    if ticker not in next_data.index:
                        continue

                    capital = min(cash * weights.iloc[j], equity * 0.2)

                    exit_idx = i + HOLD_DAYS
                    if exit_idx >= len(dates):
                        continue

                    positions.append({
                        "ticker": ticker,
                        "entry_price": next_data.loc[ticker, "Open"] * (1 + COST_RATE),
                        "exit_idx": exit_idx,
                        "capital": capital
                    })

                    cash -= capital

        # =========================
        # エクイティ
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
    # 指標計算
    # =========================
    trade_df = pd.Series(trade_logs)

    equity_df = pd.DataFrame({
        "Equity": equity_curve
    })

    if len(trade_df) == 0:
        return None

    # PF
    pf = trade_df[trade_df > 0].mean() / abs(trade_df[trade_df < 0].mean())

    # DD
    peak = equity_df["Equity"].cummax()
    dd = equity_df["Equity"] / peak - 1
    max_dd = dd.min()

    return {
        "PF": pf,
        "Trades": len(trade_df),
        "MaxDD": max_dd,
        "FinalEquity": equity_df["Equity"].iloc[-1]
    }

# =========================
# WF
# =========================
results = []

for train_years, test_year in WF_PERIODS:

    print(f"\n=== WF: Train {train_years} → Test {test_year} ===")

    train_df = df[df["Date"].dt.year.isin(train_years)]
    test_df  = df[df["Date"].dt.year == test_year]

    model = train_model(train_df)

    res = run_backtest(model, test_df)

    if res is None:
        continue

    res["Train"] = str(train_years)
    res["Test"] = test_year

    results.append(res)

# =========================
# 指標計算
# =========================
trade_df = pd.Series(trade_logs)

equity_df = pd.DataFrame({
    "Equity": equity_curve
})

if len(trade_df) == 0:
    return None

# 🔥 日次リターン
equity_df["Return"] = equity_df["Equity"].pct_change().fillna(0)

# =========================
# PF
# =========================
pf = trade_df[trade_df > 0].mean() / abs(trade_df[trade_df < 0].mean())

# =========================
# DD
# =========================
peak = equity_df["Equity"].cummax()
dd = equity_df["Equity"] / peak - 1
max_dd = dd.min()

# =========================
# 🔥 CAGR（年率）
# =========================
days = len(equity_df)
years = days / 252  # 営業日ベース
final_equity = equity_df["Equity"].iloc[-1]

cagr = final_equity ** (1 / years) - 1 if years > 0 else 0

# =========================
# 🔥 Sharpe
# =========================
sharpe = (
    equity_df["Return"].mean() / equity_df["Return"].std()
) * np.sqrt(252) if equity_df["Return"].std() != 0 else 0

return {
    "PF": pf,
    "Trades": len(trade_df),
    "MaxDD": max_dd,
    "FinalEquity": final_equity,
    "CAGR": cagr,
    "Sharpe": sharpe
}

# =========================
# 結果
# =========================
result_df = pd.DataFrame(results)

print("\n=== 最終結果（コスト込み・DD付き・固定SL） ===")
print(result_df)

print("\n=== 平均DD ===")
print(result_df["MaxDD"].mean())

print("\n=== 平均CAGR ===")
print(result_df["CAGR"].mean())

print("\n=== 平均Sharpe ===")
print(result_df["Sharpe"].mean())
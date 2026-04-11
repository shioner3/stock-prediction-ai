import pandas as pd
import numpy as np
from lightgbm import LGBMRanker

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

INITIAL_CAPITAL = 1.0
TOP_N = 5
HOLD_DAYS = 5   # 🔥 Targetと一致させる

STOP_LOSS = -0.05   # 緩める
COST_RATE = 0.0015  # 現実寄り
SLIPPAGE = 0.001

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

# =========================
# TargetClass
# =========================
def make_target_class(x):
    try:
        return pd.qcut(x, q=30, labels=False, duplicates="drop")
    except:
        return pd.cut(x, bins=min(30, len(x)), labels=False)

df["TargetClass"] = df.groupby("Date")["Target"].transform(make_target_class).astype(int)

# =========================
# モデル
# =========================
def train_model(train_df):
    group = train_df.groupby("Date").size().to_list()

    model = LGBMRanker(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
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

    data_df = data_df.dropna(subset=FEATURES).copy()

    data_df["raw_score"] = model.predict(data_df[FEATURES])
    data_df["score"] = data_df.groupby("Date")["raw_score"].rank(pct=True)

    grouped = {d: g.set_index("Ticker") for d, g in data_df.groupby("Date")}
    dates = sorted(grouped.keys())

    equity = cash = INITIAL_CAPITAL
    positions = []
    equity_curve = []

    for i, d in enumerate(dates):

        today = grouped[d]

        # 決済
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

            else:
                new_positions.append(pos)

        positions = new_positions

        # エントリー
        if i + 1 < len(dates):

            today_f = today.copy()

            # 🔥 フィルタ削除（これが重要）
            picks = today_f.sort_values("score", ascending=False).head(TOP_N)

            next_data = grouped[dates[i+1]]

            for ticker, row in picks.iterrows():

                if ticker not in next_data.index:
                    continue

                capital = cash / TOP_N

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

        # 評価
        pos_val = 0
        for pos in positions:
            if pos["ticker"] in today.index:
                price = today.loc[pos["ticker"], "Close"]
                ret = (price - pos["entry_price"]) / pos["entry_price"]
                pos_val += pos["capital"] * (1 + ret)

        equity = cash + pos_val
        equity_curve.append(equity)

    equity_df = pd.DataFrame({"Equity": equity_curve})
    equity_df["Return"] = equity_df["Equity"].pct_change().fillna(0)

    cagr = equity_df["Equity"].iloc[-1] ** (252/len(equity_df)) - 1
    sharpe = equity_df["Return"].mean() / equity_df["Return"].std() * np.sqrt(252)

    return cagr, sharpe

# =========================
# 実行
# =========================
years = sorted(df["Date"].dt.year.unique())

for i in range(3, len(years)):
    train = df[df["Date"].dt.year.isin(years[:i])]
    test  = df[df["Date"].dt.year == years[i]]

    model = train_model(train)
    cagr, sharpe = run_backtest(model, test)

    print(years[i], "CAGR:", round(cagr,3), "Sharpe:", round(sharpe,3))
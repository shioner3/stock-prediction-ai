import pandas as pd
import numpy as np
from lightgbm import LGBMRanker

# =========================
# 設定（軽量化）
# =========================
DATA_PATH = "ml_dataset.parquet"

INITIAL_CAPITAL = 1.0

TOP_N = 3
TOP_RATE = 0.005
HOLD_DAYS = 7
STOP_LOSS = -0.05

USE_MARKET_FILTER = True
N_CLASS = 30

USE_YEARS = [2023, 2024]
MAX_TICKERS = 1000

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

df = df[df["Date"].dt.year.isin(USE_YEARS)]

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
# TargetClass
# =========================
def make_target_class(x):
    try:
        return pd.qcut(x, q=N_CLASS, labels=False, duplicates="drop")
    except:
        return pd.cut(x, bins=min(N_CLASS, len(x)), labels=False)

df["TargetClass"] = df.groupby("Date")["Target"].transform(make_target_class).astype(int)

# =========================
# train/test分割
# =========================
split_date = df["Date"].quantile(0.7)

train_df = df[df["Date"] < split_date]
test_df  = df[df["Date"] >= split_date]

# =========================
# モデル
# =========================
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

# =========================
# 予測
# =========================
test_df = test_df.copy()
test_df["raw_score"] = model.predict(test_df[FEATURES])
test_df["score"] = test_df.groupby("Date")["raw_score"].rank(pct=True)

grouped = {d: g.set_index("Ticker") for d, g in test_df.groupby("Date")}
dates = sorted(grouped.keys())

# =========================
# バックテスト（SL/TP付き）
# =========================
equity = INITIAL_CAPITAL
cash = INITIAL_CAPITAL
positions = []
trade_returns = []

for i, d in enumerate(dates):

    today = grouped[d]

    # =========================
    # 決済（SL/TP + 期限）
    # =========================
    new_positions = []

    for pos in positions:

        if pos["ticker"] not in today.index:
            continue

        price = today.loc[pos["ticker"], "Close"]
        ret = (price - pos["entry_price"]) / pos["entry_price"]

        # 🔥 強制決済条件
        if ret <= STOP_LOSS or i == pos["exit_idx"]:

            exit_price = today.loc[pos["ticker"], "Open"]
            final_ret = (exit_price - pos["entry_price"]) / pos["entry_price"]

            cash += pos["capital"] * (1 + final_ret)
            trade_returns.append(final_ret)

        else:
            new_positions.append(pos)

    positions = new_positions

    # =========================
    # エントリー
    # =========================
    if i + 1 < len(dates):

        next_data = grouped[dates[i+1]]
        today_f = today.copy()

        if USE_MARKET_FILTER:
            today_f = today_f[today_f["Market_Trend"] > 0.003]

        today_f = today_f[today_f["Trend_5_z"] > 0.8]
        today_f = today_f[today_f["score"] >= (1 - TOP_RATE)]

        if len(today_f) > 0:

            picks = today_f.sort_values("score", ascending=False).head(TOP_N)

            weights = (picks["score"] ** 2) * (1 + picks["Trend_5_z"].clip(0, 2))
            weights = weights / weights.sum()

            for j, (ticker, row) in enumerate(picks.iterrows()):

                if ticker not in next_data.index:
                    continue

                capital = cash * weights.iloc[j]

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
# 結果
# =========================
trade_df = pd.Series(trade_returns)

print("\n=== 簡易バックテスト結果（SL/TPあり） ===")

print(f"Trades: {len(trade_df)}")

if len(trade_df) > 0:
    print(f"WinRate: {(trade_df > 0).mean():.3f}")
    print(f"AvgReturn: {trade_df.mean():.4f}")
    print(f"PF: {trade_df[trade_df>0].mean() / abs(trade_df[trade_df<0].mean()):.3f}")
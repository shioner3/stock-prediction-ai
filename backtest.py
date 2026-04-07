import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from tqdm import tqdm

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

INITIAL_CAPITAL = 1.0
TOP_N = 3
HOLD_DAYS = 7
RETRAIN_SPAN = 20

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
    "Gap","Volatility_change","Momentum_acc",
    "Return_1_rank","Return_3_rank",
    "Volume_ratio_rank","Trend_5_z_rank","HL_range_rank",
    "Market_Z","Market_Trend",
    "DayOfWeek"
]

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df = df.sort_values("Date").reset_index(drop=True)
dates = df["Date"].unique()

# =========================
# 状態管理
# =========================
capital = INITIAL_CAPITAL
equity_curve = []

positions = []
model = None

# 分析用ログ
trade_logs = []
daily_logs = []

# =========================
# バックテスト
# =========================
for i in tqdm(range(len(dates))):

    current_date = dates[i]

    # =========================
    # ① 決済
    # =========================
    new_positions = []
    daily_return = 0

    for pos in positions:
        pos["days"] += 1

        if pos["days"] >= HOLD_DAYS:
            exit_row = df[
                (df["Date"] == current_date) &
                (df["Ticker"] == pos["Ticker"])
            ]

            if len(exit_row) > 0:
                exit_price = exit_row["Close"].values[0]
                ret = (exit_price / pos["entry_price"]) - 1

                pnl = ret * pos["weight"]
                daily_return += pnl

                # 🔥 トレードログ
                trade_logs.append({
                    "EntryDate": pos["entry_date"],
                    "ExitDate": current_date,
                    "Ticker": pos["Ticker"],
                    "Return": ret,
                    "Weight": pos["weight"],
                    "Score": pos["score"],
                    "Trend": pos["trend"]
                })
        else:
            new_positions.append(pos)

    positions = new_positions

    # =========================
    # ② 資産更新
    # =========================
    capital *= (1 + daily_return)
    equity_curve.append(capital)

    daily_logs.append({
        "Date": current_date,
        "Return": daily_return,
        "Capital": capital
    })

    # =========================
    # ③ 学習
    # =========================
    if i < 100:
        continue

    if (model is None) or (i % RETRAIN_SPAN == 0):

        train_df = df[df["Date"] < current_date].dropna(subset=FEATURES + ["Target"])
        if len(train_df) < 1000:
            continue

        model = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(train_df[FEATURES], train_df["Target"])

    # =========================
    # ④ 予測
    # =========================
    today_df = df[df["Date"] == current_date].dropna(subset=FEATURES)

    if len(today_df) == 0:
        continue

    today_df = today_df.copy()
    today_df["raw_score"] = model.predict(today_df[FEATURES])
    today_df["score"] = today_df["raw_score"].rank(pct=True)

    today_df = today_df[today_df["score"] > 0.8]
    today_df = today_df.sort_values("score", ascending=False).head(TOP_N)

    if len(today_df) == 0:
        continue

    # =========================
    # ⑤ エントリー
    # =========================
    weights = (1 + today_df["Trend_5_z"].clip(-1, 1))
    weights = weights / weights.sum()

    for (_, row), w in zip(today_df.iterrows(), weights):
        positions.append({
            "Ticker": row["Ticker"],
            "entry_price": row["Close"],
            "entry_date": current_date,
            "weight": w,
            "days": 0,
            "score": row["score"],
            "trend": row["Trend_5_z"]
        })

# =========================
# 結果計算
# =========================
equity_curve = pd.Series(equity_curve)
returns = equity_curve.pct_change().dropna()

cagr = (equity_curve.iloc[-1]) ** (252 / len(equity_curve)) - 1
sharpe = returns.mean() / returns.std() * np.sqrt(252)
max_dd = (equity_curve / equity_curve.cummax() - 1).min()

# =========================
# 🔥 分析
# =========================
trades = pd.DataFrame(trade_logs)
daily = pd.DataFrame(daily_logs)

print("\n===== Backtest Result =====")
print(f"CAGR: {cagr:.2%}")
print(f"Sharpe: {sharpe:.2f}")
print(f"MaxDD: {max_dd:.2%}")

# =========================
# 勝率・PF
# =========================
win_rate = (trades["Return"] > 0).mean()
avg_win = trades[trades["Return"] > 0]["Return"].mean()
avg_loss = trades[trades["Return"] <= 0]["Return"].mean()
pf = trades[trades["Return"] > 0]["Return"].sum() / abs(trades[trades["Return"] <= 0]["Return"].sum())

print("\n=== Trade Stats ===")
print(f"Trades: {len(trades)}")
print(f"WinRate: {win_rate:.2%}")
print(f"AvgWin: {avg_win:.4f}")
print(f"AvgLoss: {avg_loss:.4f}")
print(f"PF: {pf:.2f}")

# =========================
# 最大連敗
# =========================
loss_streak = (trades["Return"] <= 0).astype(int)
max_losing_streak = (loss_streak.groupby((loss_streak != loss_streak.shift()).cumsum()).cumsum()).max()

print("\nMax Losing Streak:", max_losing_streak)

# =========================
# 月次リターン
# =========================
daily["Month"] = pd.to_datetime(daily["Date"]).dt.to_period("M")
monthly = daily.groupby("Month")["Return"].sum()

print("\n=== Monthly Return ===")
print(monthly.tail(12))

# =========================
# スコア別リターン
# =========================
trades["score_bin"] = pd.qcut(trades["Score"], 10, labels=False)
score_analysis = trades.groupby("score_bin")["Return"].mean()

print("\n=== Score別リターン ===")
print(score_analysis)

# =========================
# レジーム別
# =========================
trades["trend_bin"] = pd.qcut(trades["Trend"], 5, labels=False)
trend_analysis = trades.groupby("trend_bin")["Return"].mean()

print("\n=== Trend別リターン ===")
print(trend_analysis)

# =========================
# DD期間
# =========================
dd = equity_curve / equity_curve.cummax() - 1
dd_duration = (dd < 0).astype(int)
max_dd_duration = dd_duration.groupby((dd_duration != dd_duration.shift()).cumsum()).cumsum().max()

print("\nMax DD Duration:", max_dd_duration)
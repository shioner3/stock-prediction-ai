import pandas as pd
import numpy as np
from lightgbm import LGBMRanker

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

INITIAL_CAPITAL = 1.0
TOP_N = 5
HOLD_DAYS = 10

TREND_TH = 1.0
TOP_RATE = 0.014

STOP_LOSS = -0.02
COST_RATE = 0.0025
SLIPPAGE = 0.002

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

# =========================
# 特徴量
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
# Target
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
    train_df = train_df.sort_values("Date")
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
# 可変ホールド
# =========================
def calc_hold_days(row):
    hold = (
        HOLD_DAYS
        + int(row["Trend_5_z"] * 2)
        + int((1 - row["TrendVol"]) * 5)
        + int((row["DD_5"] + 0.1) * 5)
    )
    return max(5, min(20, hold))

# =========================
# 最大連敗
# =========================
def calc_max_losing_streak(trades):
    streak = max_streak = 0
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

    data_df = data_df.dropna(subset=FEATURES).copy()

    # =========================
    # 🔥 正しいスコア処理（rank → shift）
    # =========================
    data_df["raw_score"] = model.predict(data_df[FEATURES])
    data_df["score"] = data_df.groupby("Date")["raw_score"].rank(pct=True)

    # 🔥 1日遅延（リーク防止）
    data_df["score_shift"] = data_df.groupby("Ticker")["score"].shift(1)

    grouped = {d: g.set_index("Ticker") for d, g in data_df.groupby("Date")}
    dates = sorted(grouped.keys())

    equity = cash = INITIAL_CAPITAL
    positions = []

    trade_logs = []
    equity_curve = []

    for i, d in enumerate(dates):

        today = grouped[d]

        # =========================
        # 決済（intradayストップロス）
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
                final_ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                final_ret -= COST_RATE * 2

                cash += pos["capital"] * (1 + final_ret)
                trade_logs.append(final_ret)

            else:
                new_positions.append(pos)

        positions = new_positions

        # =========================
        # エントリー（昨日スコア）
        # =========================
        if i + 1 < len(dates):

            today_f = today.copy()

            # 🔥 遅延スコア使用
            today_f = today_f.dropna(subset=["score_shift"])

            # フィルタ
            today_f = today_f[today_f["Market_Trend"] > 0]
            today_f = today_f[today_f["Trend_5_z"] > TREND_TH]

            # 🔥 上位率（rank復活）
            today_f = today_f[today_f["score_shift"] >= (1 - TOP_RATE)]

            if len(today_f) > 0:

                today_f["adj_score"] = (
                    today_f["score_shift"]
                    * (1 + today_f["Trend_5_z"].clip(0, 2))
                    * (1 - today_f["TrendVol"].clip(0, 1))
                    * (1 + today_f["DD_5"].clip(-0.2, 0.2))
                )

                picks = today_f.sort_values("adj_score", ascending=False).head(TOP_N)

                weights = picks["adj_score"]
                weights = (
                    (weights / weights.sum()).values
                    if weights.sum() > 0
                    else np.ones(len(weights)) / len(weights)
                )

                next_data = grouped[dates[i+1]]

                for j, (ticker, row) in enumerate(picks.iterrows()):

                    if ticker not in next_data.index:
                        continue

                    capital = min(cash * weights[j], equity * 0.15)

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

        # =========================
        # 評価額
        # =========================
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

    equity_df = pd.DataFrame({"Equity": equity_curve})
    equity_df["Return"] = equity_df["Equity"].pct_change().fillna(0)

    peak = equity_df["Equity"].cummax()
    max_dd = (equity_df["Equity"] / peak - 1).min()

    years = len(equity_df) / 252
    cagr = equity_df["Equity"].iloc[-1] ** (1 / years) - 1

    sharpe = (
        equity_df["Return"].mean()
        / equity_df["Return"].std()
        * np.sqrt(252)
    ) if equity_df["Return"].std() != 0 else 0

    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "LosingStreak": calc_max_losing_streak(trade_logs),
        "Trades": len(trade_logs)
    }

# =========================
# ウォークフォワード
# =========================
results = []

years = sorted(df["Date"].dt.year.unique())

for i in range(3, len(years)):

    train_years = years[:i]
    test_year = years[i]

    print(f"Train {train_years} → Test {test_year}")

    train_df = df[df["Date"].dt.year.isin(train_years)]
    test_df  = df[df["Date"].dt.year == test_year]

    model = train_model(train_df)
    res = run_backtest(model, test_df)

    if res:
        res["Year"] = test_year
        results.append(res)

# =========================
# 結果
# =========================
result_df = pd.DataFrame(results)

if len(result_df) == 0:
    print("⚠️ 結果なし")
else:
    print("\n=== Yearly Performance ===")
    print(result_df[["Year","CAGR","Sharpe","MaxDD"]])

    print("\n=== Summary ===")
    print(f"CAGR        : {result_df['CAGR'].mean():.3f}")
    print(f"Sharpe      : {result_df['Sharpe'].mean():.3f}")
    print(f"MaxDD       : {result_df['MaxDD'].mean():.3f}")
    print(f"LosingStreak: {result_df['LosingStreak'].mean():.1f}")
    print(f"Trades      : {result_df['Trades'].mean():.1f}")
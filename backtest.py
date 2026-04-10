import pandas as pd
import numpy as np
from lightgbm import LGBMRanker

# =========================
# 戦略概要
# =========================
# ・LightGBM Rankerで銘柄ランキング
# ・上位1.4%のみエントリー
# ・トレンド強い銘柄を優先
# ・最大5銘柄まで保有
# ・1銘柄あたり最大20%投資
# ・ストップロス -2%

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
# データ読み込み
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
# ターゲット作成
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

    data_df["score"] = (
        data_df.groupby("Date")[FEATURES]
        .apply(lambda x: model.predict(x))
        .reset_index(level=0, drop=True)
    )
    data_df["score"] = data_df.groupby("Date")["score"].rank(pct=True)

    grouped = {d: g.set_index("Ticker") for d, g in data_df.groupby("Date")}
    dates = sorted(grouped.keys())

    equity = cash = INITIAL_CAPITAL
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

            # フィルタ
            today_f = today_f[today_f["Market_Trend"] > 0]
            today_f = today_f[today_f["Trend_5_z"] > TREND_TH]
            today_f = today_f[today_f["score"] >= (1 - TOP_RATE)]

            # スコア統合
            today_f["adj_score"] = (
                today_f["score"]
                * (1 + today_f["Trend_5_z"].clip(0, 2))
                * (1 - today_f["TrendVol"].clip(0, 1))
                * (1 + today_f["DD_5"].clip(-0.2, 0.2))
                * (1 + today_f["Market_Trend"].clip(0, 0.02))
            )

            picks = today_f.sort_values("adj_score", ascending=False).head(TOP_N)

            if len(picks) > 0:

                w = picks["adj_score"]
                w = (w / w.sum()).values if w.sum() > 0 else np.ones(len(w)) / len(w)

                for j, (ticker, row) in enumerate(picks.iterrows()):

                    if ticker not in next_data.index:
                        continue

                    capital = min(cash * w[j], equity * 0.2)

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
        pos_val = sum(
            pos["capital"] * (
                1 + (today.loc[pos["ticker"], "Close"] - pos["entry_price"]) / pos["entry_price"]
            )
            for pos in positions if pos["ticker"] in today.index
        )

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

    sharpe = equity_df["Return"].mean() / equity_df["Return"].std() * np.sqrt(252)

    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "LosingStreak": calc_max_losing_streak(trade_logs),
        "Trades": len(trade_logs)
    }

# =========================
# ウォークフォワード検証
# =========================
results = []

years = sorted(df["Date"].dt.year.unique())

for i in range(3, len(years)):

    train_years = years[:i]
    test_year = years[i]

    print(f"Train {train_years} → Test {test_year}")

    model = train_model(df[df["Date"].dt.year.isin(train_years)])
    res = run_backtest(model, df[df["Date"].dt.year == test_year])

    if res:
        res["Year"] = test_year
        results.append(res)

# =========================
# 結果表示
# =========================
result_df = pd.DataFrame(results)

print("\n=== Yearly Performance ===")
print(result_df[["Year","CAGR","Sharpe","MaxDD"]])

print("\n=== Summary ===")
print(f"CAGR        : {result_df['CAGR'].mean():.3f}")
print(f"Sharpe      : {result_df['Sharpe'].mean():.3f}")
print(f"MaxDD       : {result_df['MaxDD'].mean():.3f}")
print(f"LosingStreak: {result_df['LosingStreak'].mean():.1f}")
print(f"Trades      : {result_df['Trades'].mean():.1f}")
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

INITIAL_CAPITAL = 1.0
MAX_POSITIONS = 5
ALPHA = 2.0

TOP_RATE = 0.01
TREND_TH = 0.0
HOLD_DAYS = 7

STOP_LOSS = -0.08
TAKE_PROFIT = 0.15
EXIT_RANK_TH = 0.6

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
    "Trend_5_z","Trend_10_z"
]

years = sorted(df["Date"].dt.year.unique())

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
    test_df["score"] = model.predict(test_df[FEATURES])
    test_df["score_rank"] = test_df["score"].rank(pct=True)

    grouped = {d: g for d, g in test_df.groupby("Date")}
    dates = sorted(grouped.keys())
    date_index = {d: i for i, d in enumerate(dates)}

    equity = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    equity_curve = []

    positions = []

    trade_count = 0  # 🔥追加

    for d in dates:

        today = grouped[d]

        # =========================
        # 決済（EXITロジック）
        # =========================
        new_positions = []

        for pos in positions:

            cur = today[today["Ticker"] == pos["ticker"]]

            if cur.empty:
                continue

            price = cur["Open"].iloc[0]
            current_return = (price - pos["entry_price"]) / pos["entry_price"]

            exit_flag = False

            # 🔥 損切り
            if current_return < STOP_LOSS:
                exit_flag = True

            # 🔥 利確
            elif current_return > TAKE_PROFIT:
                exit_flag = True

            # 🔥 スコア弱化
            elif "score_rank" in cur.columns:
                if cur["score_rank"].iloc[0] < EXIT_RANK_TH:
                    exit_flag = True

            # 🔥 時間切れ
            elif d == pos["exit_date"]:
                exit_flag = True

            if exit_flag:
                cash += pos["capital"] * (1 + current_return)
                trade_count += 1
            else:
                new_positions.append(pos)

        positions = new_positions

        # =========================
        # エントリー
        # =========================
        if date_index[d] + 1 < len(dates):

            available = MAX_POSITIONS - len(positions)

            if available > 0 and cash > 0:

                next_day = dates[date_index[d] + 1]
                next_data = grouped[next_day]

                today_f = today[
                    (today["Trend_5_z"] > TREND_TH) &
                    (today["score"] > today["score"].quantile(0.9))
                ]

                if len(today_f) > 0:

                    TOP_K = max(3, int(len(today_f) * TOP_RATE))
                    picks = today_f.nlargest(TOP_K, "score")
                    picks = picks.head(available)

                    scores = picks["score"].values
                    scores = scores - scores.mean()
                    scores = np.clip(scores, 0, None)

                    if scores.sum() > 0:

                        weights = scores ** ALPHA
                        weights /= weights.sum()

                        for i, row in enumerate(picks.itertuples()):

                            ticker = row.Ticker

                            if any(p["ticker"] == ticker for p in positions):
                                continue

                            next_row = next_data[next_data["Ticker"] == ticker]
                            if next_row.empty:
                                continue

                            capital = cash * weights[i]
                            capital = min(capital, equity * 0.2)

                            if capital <= 0:
                                continue

                            entry_price = next_row["Open"].iloc[0]

                            positions.append({
                                "ticker": ticker,
                                "entry_price": entry_price,
                                "exit_date": next_day + pd.Timedelta(days=HOLD_DAYS),
                                "capital": capital
                            })

                            cash -= capital

        # =========================
        # 評価額
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

    equity_curve = pd.Series(equity_curve)
    returns = equity_curve.pct_change().dropna()

    if len(equity_curve) < 50:
        return None

    CAGR = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
    Sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
    MaxDD = (equity_curve / equity_curve.cummax() - 1).min()

    return CAGR, Sharpe, MaxDD, trade_count

# =========================
# 実行
# =========================
all_metrics = []

for y in years:
    if y < 2022:
        continue

    train_df = df[df["Date"].dt.year < y]
    test_df = df[df["Date"].dt.year == y]

    res = run_backtest(train_df, test_df)

    if res is not None:
        all_metrics.append(res)

if len(all_metrics) > 0:

    cagr = np.mean([m[0] for m in all_metrics])
    sharpe = np.mean([m[1] for m in all_metrics])
    mdd = np.mean([m[2] for m in all_metrics])
    trades = np.sum([m[3] for m in all_metrics])

    print("\n=== RESULT ===")
    print(f"CAGR  : {cagr:.4f}")
    print(f"Sharpe: {sharpe:.4f}")
    print(f"MaxDD : {mdd:.4f}")
    print(f"Trades: {trades}")
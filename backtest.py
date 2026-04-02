import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

FEATURES = [
    "Return_1","Return_3","Return_5",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio","Volume_accel",
    "HL_range",
    "EMA_gap",
    "Momentum_5","Momentum_10",
    "ATR_ratio",
    "RSI"
]

TARGET = "Target"

INITIAL_CAPITAL = 1.0

THRESHOLD = 0.28
HOLD_DAYS = 7
STOP_LOSS = -0.02
TAKE_PROFIT = 0.08

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

# 🔥 市場データ作成（疑似TOPIX）
market_df = df.groupby("Date")["Return_1"].mean().rolling(5).mean()

# =========================
# ウォークフォワード
# =========================
years = sorted(df["Date"].dt.year.unique())
results = []

for test_year in years:

    if test_year < 2022:
        continue

    print(f"\n=== WALK FORWARD: {test_year} ===")

    train_df = df[df["Date"].dt.year < test_year]
    test_df = df[df["Date"].dt.year == test_year].copy()

    if len(test_df) == 0:
        continue

    # =========================
    # 学習
    # =========================
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(train_df[FEATURES], train_df[TARGET])

    # =========================
    # 予測
    # =========================
    test_df["pred"] = model.predict_proba(test_df[FEATURES])[:, 1]

    # =========================
    # ハイブリッド
    # =========================
    def make_hybrid_score(df):
        df = df.copy()
        df["mom_rank"] = df["Return_5"].rank(pct=True)
        df["trend_rank"] = df["EMA_gap"].rank(pct=True)
        df["vol_rank"] = (-df["Volatility"]).rank(pct=True)

        df["hybrid_score"] = (
            0.5 * df["mom_rank"] +
            0.3 * df["trend_rank"] +
            0.2 * df["vol_rank"]
        )
        return df

    dates = sorted(test_df["Date"].unique())
    date_index = {d: i for i, d in enumerate(dates)}

    equity = INITIAL_CAPITAL
    equity_curve = []
    positions = []
    trade_count = 0

    for d in dates:

        today = test_df[test_df["Date"] == d]
        daily_pnl = 0

        # =========================
        # 決済
        # =========================
        new_positions = []

        for pos in positions:
            cur = today[today["Ticker"] == pos["ticker"]]

            if cur.empty:
                new_positions.append(pos)
                continue

            price = cur["Close"].iloc[0]
            ret = (price - pos["entry_price"]) / pos["entry_price"]

            if ret < STOP_LOSS or ret > TAKE_PROFIT or d >= pos["exit_date"]:
                pnl = pos["capital"] * ret
                daily_pnl += pnl
            else:
                new_positions.append(pos)

        positions = new_positions

        # =========================
        # エントリー候補
        # =========================
        today_f = today.copy()
        today_f = today_f[today_f["pred"] > THRESHOLD]
        today_f = today_f[today_f["EMA_gap"] > 0.01]

        if not today_f.empty:

            # 🔥 市場（改善）
            if d not in market_df.index:
                equity += daily_pnl
                equity_curve.append(equity)
                continue

            market = market_df.loc[d]

            # 🔥 銘柄側補助指標
            market_trend = today_f["EMA_gap"].mean()
            market_pred_mean = today_f["pred"].mean()

            # =========================
            # 🔥 フィルタ（超重要）
            # =========================

            # 完全回避（下げ相場）
            if market < 0:
                equity += daily_pnl
                equity_curve.append(equity)
                continue

            # 弱い日回避
            if market_pred_mean < 0.30:
                equity += daily_pnl
                equity_curve.append(equity)
                continue

            # トレンド弱い
            if market_trend < 0:
                equity += daily_pnl
                equity_curve.append(equity)
                continue

            # =========================
            # ポジション調整
            # =========================
            if market < 0.005:
                weight_cap = 0.3
                top_n = 1
            else:
                weight_cap = 0.4
                top_n = 2

            # =========================
            # 銘柄選定
            # =========================
            today_f = make_hybrid_score(today_f)
            picks = today_f.sort_values("hybrid_score", ascending=False).head(top_n)

            total_pred = picks["pred"].sum()

            if total_pred > 0:

                invested = sum([p["capital"] for p in positions])
                free_cash = equity - invested

                if d
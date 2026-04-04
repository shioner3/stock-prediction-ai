import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from hmmlearn.hmm import GaussianHMM

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

FEATURES = [
    "Return_1","Return_3",
    "Rank_Return_1","Rank_Volume",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio",
    "HL_range",
    "RSI",
    "Rel_Return_1",
    "Trend_5_z",
    "Trend_10_z",
    "Volatility_z",
    "Volume_ratio_z",
    "Market_Return_z"
]

TARGET = "Target"

INITIAL_CAPITAL = 1.0
HOLD_DAYS = 7

STOP_LOSS = -0.03
TAKE_PROFIT = 0.10

THRESHOLD = 0.52

# 🆕 追加
CRASH_FILTER = -0.02
TOP_N = 2

# =========================
# 手作りレジーム
# =========================
MARKET_FILTER = -0.003
MARKET_MA_WINDOW = 20

# =========================
# HMM設定
# =========================
HMM_FEATURES = ["Return_1", "Volatility", "Volume_ratio"]
HMM_SKIP = 0.75
HMM_WEAK = 0.55

# =========================
# DD STOP
# =========================
MAX_DRAWDOWN = -0.20

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

# =========================
# 市場データ（HMM）
# =========================
market = df.groupby("Date")[HMM_FEATURES].mean().reset_index()

hmm = GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=200,
    random_state=42
)

hmm.fit(market[HMM_FEATURES])
proba = hmm.predict_proba(market[HMM_FEATURES])

# 状態意味付け
state_ret = {}
for i in range(3):
    mask = proba[:, i] > 0.5
    state_ret[i] = market["Return_1"][mask].mean() if mask.sum() > 0 else 0

sorted_states = sorted(state_ret.items(), key=lambda x: x[1])
DOWN_STATE = sorted_states[0][0]

market["down_p"] = proba[:, DOWN_STATE]

# =========================
# 学習
# =========================
train_df = df[df["Date"].dt.year <= 2021]

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    random_state=42
)

print("Training model...")
model.fit(train_df[FEATURES], train_df[TARGET])
print("Done.")

# =========================
# バックテスト
# =========================
def run_backtest(test_df):

    test_df = test_df.copy()
    test_df["pred"] = model.predict_proba(test_df[FEATURES])[:, 1]

    dates = sorted(test_df["Date"].unique())
    date_index = {d: i for i, d in enumerate(dates)}

    market_map = market.set_index("Date")[["down_p"]]
    test_df = test_df.merge(market_map, left_on="Date", right_index=True, how="left")

    equity = INITIAL_CAPITAL
    equity_curve = []
    positions = []

    peak_equity = INITIAL_CAPITAL
    trading_stopped = False

    for i, d in enumerate(dates):

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
                daily_pnl += pos["capital"] * ret
            else:
                new_positions.append(pos)

        positions = new_positions
        equity += daily_pnl

        # =========================
        # DD管理
        # =========================
        peak_equity = max(peak_equity, equity)
        dd = (equity / peak_equity) - 1

        if dd < MAX_DRAWDOWN:
            trading_stopped = True

        if trading_stopped and dd > -0.05:
            trading_stopped = False

        if trading_stopped:
            equity_curve.append(equity)
            continue

        # =========================
        # 🚨 クラッシュフィルタ（最強）
        # =========================
        market_ret = today["Return_1"].mean()

        if market_ret < CRASH_FILTER:
            equity_curve.append(equity)
            continue

        # =========================
        # 手作りレジーム
        # =========================
        if i >= MARKET_MA_WINDOW:
            past_dates = dates
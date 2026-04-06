import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# =========================
# 設定
# =========================
DATA_PATH = "ml_dataset.parquet"

INITIAL_CAPITAL = 1.0
MAX_POSITIONS = 10
HOLD_DAYS = 10

TREND_TH = 0.0
N_BINS = 10

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
    "Trend_5_z","Trend_10_z",
    "Gap","Volatility_change","Volume_spike","Momentum_acc"
]

# =========================
# レジーム
# =========================
market = df.groupby("Date")["Return_1"].mean()
market_smooth = market.rolling(20).mean()

df["Market_Smooth"] = df["Date"].map(market_smooth)

df["Regime"] = np.where(
    df["Market_Smooth"] > 0.001, "up",
    np.where(df["Market_Smooth"] < -0.001, "down", "range")
)

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
# スコア帯最適化
# =========================
def optimize_score_range(train_df, model):

    df_tmp = train_df.copy()
    df_tmp["raw_score"] = model.predict(df_tmp[FEATURES])

    _, bins = pd.qcut(
        df_tmp["raw_score"],
        N_BINS,
        retbins=True,
        duplicates="drop"
    )

    df_tmp["bin"] = pd.cut(
        df_tmp["raw_score"],
        bins=bins,
        include_lowest=True
    )

    stats = df_tmp.groupby("bin")["Target"].mean()

    # 上位2bin
    good_bins = stats.sort_values().tail(2).index

    print("\n=== SCORE BIN OPT ===")
    print(stats)
    print("USE TOP BINS:", list(good_bins))

    return bins, good_bins

# =========================
# バックテスト
# =========================
def run_backtest(train_df, test_df):

    model = train_model(train_df)

    bins, good_bins = optimize_score_range(train_df, model)

    test_df = test_df.copy()
    test_df["raw_score"] = model.predict(test_df[FEATURES])
    test_df["score"] = test_df.groupby("Date")["raw_score"].rank(pct=True)

    # bin適用
    test_df["bin"] = pd.cut(
        test_df["raw_score"],
        bins=bins,
        include_lowest=True
    )

    # binランク化
    bin_order = {b: i for i, b in enumerate(sorted(good_bins))}
    test_df["bin_rank"] = test_df["bin"].map(bin_order)

    dates = sorted(test_df["Date"].unique())
    grouped = {d: g for d, g in test_df.groupby("Date")}

    equity = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    equity_curve = []

    positions = []

    for i, d in enumerate(dates):

        today = grouped[d]

        # =========================
        # 決済
        # =========================
        new_positions = []

        for pos in positions:
            if i == pos["exit_idx"]:

                cur = today[today["Ticker"] == pos["ticker"]]
                if cur.empty:
                    continue

                exit_price = cur["Open"].iloc[0]
                ret = (exit_price - pos["entry_price
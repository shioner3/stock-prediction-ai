import pandas as pd
import numpy as np
import pickle

# =========================
# 設定
# =========================
DATA_PATH = "stock_data/ml_dataset.parquet"

MODEL_PATH = "stock_data/lgbm_ranker.pkl"

INITIAL_CAPITAL = 1.0

TOP_N = 3
MAX_POSITIONS = 5

HOLD_DAYS = 5

STOP_LOSS = -0.07

SLIPPAGE = 0.002
COMMISSION = 0.001

TEST_START_DATE = "2024-01-01"

# =========================
# 特徴量
# =========================
FEATURES = [

    "close_ma5_ratio",
    "close_ma25_ratio",
    "ma25_slope",
    "high_break_20d",

    "return_5d",
    "return_20d",
    "relative_strength_20d",
    "industry_rs_rank",

    "volume_ratio_5d",
    "volume_ratio_20d",
    "volume_zscore",

    "atr_ratio",
    "bb_width",
    "range_compression_5d",

    "nikkei_return_5d",
    "topix_trend",
    "growth_index_strength",

    "return_rank_daily",
    "volume_rank_daily",
    "volatility_rank",
    "rs_rank_cross_section",

    "upper_shadow_ratio",
    "gap_up_ratio",
    "bb_position"
]

# =========================
# データ読み込み
# =========================
print("Loading dataset...")

df = pd.read_parquet(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Date", "Ticker"]
).reset_index(drop=True)

# =========================
# テスト期間
# =========================
df = df[
    df["Date"] >= TEST_START_DATE
].copy()

# =========================
# モデル読み込み
# =========================
print("Loading model...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# =========================
# prediction
# =========================
print("Predicting scores...")

df["pred_score"] = model.predict(
    df[FEATURES]
)

# =========================
# 日次順位
# =========================
df["pred_rank"] = (
    df.groupby("Date")["pred_score"]
    .rank(
        ascending=False,
        method="first"
    )
)

# =========================
# フィルタ
# =========================
print("Applying filters...")

df = df[
    df["return_5d"] < 0.25
]

df = df[
    df["gap_up_ratio"] < 0.08
]

df = df[
    df["upper_shadow_ratio"] < 0.04
]

df = df[
    df["bb_position"] < 0.95
]

df = df[
    df["atr_ratio"] < 0.12
]

df = df[
    df["volume_ratio_20d"] > 0.3
]

# =========================
# target_return clip
# =========================
print("Applying return clip...")

df["target_return"] = (
    df["target_return"]
    .clip(-0.3, 0.5)
)

# =========================
# 候補
# =========================
candidate_df = df[
    df["pred_rank"] <= TOP_N
].copy()

# =========================
# 日付一覧
# =========================
dates = sorted(
    candidate_df["Date"].unique()
)

# =========================
# バックテスト
# =========================
print("Running realistic backtest...")

capital = INITIAL_CAPITAL

equity_curve = []

daily_returns_list = []

positions = []

trade_log = []

# =========================
# メインループ
# =========================
for current_date in dates:

    # =====================
    # 保有ポジション処理
    # =====================
    realized_returns = []

    remaining_positions = []

    for pos in positions:

        hold_days = (
            current_date - pos["entry_date"]
        ).days

        # =================
        # exit
        # =================
        if hold_days >= HOLD_DAYS:

            ret = pos["return"]

            # 損切り
            ret = max(ret, STOP_LOSS)

            # コスト
            ret -= (
                SLIPPAGE
                + COMMISSION
            )

            realized_returns.append(ret)

            trade_log.append({
                "Date": current_date,
                "Ticker": pos["Ticker"],
                "Return": ret
            })

        else:
            remaining_positions.append(pos)

    positions = remaining_positions

    # =====================
    # 空き枠
    # =====================
    slots = (
        MAX_POSITIONS
        - len(positions)
    )

    # =====================
    # 新規エントリー
    # =====================
    if slots > 0:

        day_candidates = candidate_df[
            candidate_df["Date"]
            == current_date
        ].sort_values(
            "pred_rank"
        )

        entries = day_candidates.head(slots)

        for _, row in entries.iterrows():

            positions.append({

                "Ticker": row["Ticker"],

                # 翌日始値エントリー前提
                "entry_date": current_date,

                "return": row["target_return"]
            })

    # =====================
    # 日次リターン
    # =====================
    if len(realized_returns) > 0:

        daily_ret = np.mean(
            realized_returns
        )

    else:
        daily_ret = 0

    capital *= (
        1 + daily_ret
    )

    daily_returns_list.append({
        "Date": current_date,
        "Return": daily_ret,
        "Capital": capital
    })

    equity_curve.append(capital)

# =========================
# DataFrame化
# =========================
daily_df = pd.DataFrame(
    daily_returns_list
)

daily_df["Date"] = pd.to_datetime(
    daily_df["Date"]
)

daily_df = daily_df.set_index(
    "Date"
)

daily_returns = daily_df["Return"]

equity_curve = daily_df["Capital"]

# =========================
# CAGR
# =========================
days = (
    equity_curve.index.max()
    - equity_curve.index.min()
).days

years = days / 365

final_value = equity_curve.iloc[-1]

if years > 0:

    cagr = (
        final_value ** (1 / years)
        - 1
    )

else:
    cagr = 0

# =========================
# Sharpe
# =========================
if daily_returns.std() > 0:

    sharpe = (
        daily_returns.mean()
        / daily_returns.std()
    ) * np.sqrt(252)

else:
    sharpe = 0

# =========================
# Drawdown
# =========================
rolling_max = (
    equity_curve.cummax()
)

drawdown = (
    equity_curve
    / rolling_max
    - 1
)

max_dd = drawdown.min()

# =========================
# Calmar
# =========================
if abs(max_dd) > 0:

    calmar = (
        cagr / abs(max_dd)
    )

else:
    calmar = 0

# =========================
# 勝率
# =========================
trade_df = pd.DataFrame(trade_log)

if len(trade_df) > 0:

    win_rate = (
        (trade_df["Return"] > 0)
        .mean()
    )

    avg_return = (
        trade_df["Return"]
        .mean()
    )

else:

    win_rate = 0
    avg_return = 0

# =========================
# 年別
# =========================
yearly_result = []

for year, group in daily_returns.groupby(
    daily_returns.index.year
):

    yearly_curve = (
        (1 + group)
        .cumprod()
    )

    yearly_return = (
        yearly_curve.iloc[-1]
        - 1
    )

    if group.std() > 0:

        yearly_sharpe = (
            group.mean()
            / group.std()
        ) * np.sqrt(252)

    else:
        yearly_sharpe = 0

    yearly_result.append({

        "Year": year,
        "Return": yearly_return,
        "Sharpe": yearly_sharpe
    })

yearly_df = pd.DataFrame(
    yearly_result
)

# =========================
# 月別
# =========================
monthly_returns = (
    daily_returns
    .resample("ME")
    .apply(
        lambda x:
        (1 + x).prod() - 1
    )
)

# =========================
# 表示
# =========================
print("\n=== BACKTEST RESULT ===")

print(f"CAGR        : {cagr:.4f}")

print(f"Sharpe      : {sharpe:.4f}")

print(f"Calmar      : {calmar:.4f}")

print(f"MaxDD       : {max_dd:.4f}")

print(f"\nFinal Capital : {final_value:.4f}")

print(f"\nTrades        : {len(trade_df)}")

print(f"Win Rate      : {win_rate:.4f}")

print(f"Average Return: {avg_return:.4f}")

# =========================
# 年別
# =========================
print("\n=== YEARLY PERFORMANCE ===")

print(yearly_df)

# =========================
# 月別
# =========================
print("\n=== MONTHLY RETURNS ===")

print(
    monthly_returns.tail(12)
)

# =========================
# 保存
# =========================
equity_curve_df = pd.DataFrame({

    "Date": equity_curve.index,
    "Equity": equity_curve.values
})

equity_curve_df.to_csv(
    "stock_data/equity_curve.csv",
    index=False
)

yearly_df.to_csv(
    "stock_data/yearly_result.csv",
    index=False
)

monthly_returns.to_csv(
    "stock_data/monthly_returns.csv"
)

trade_df.to_csv(
    "stock_data/trade_log.csv",
    index=False
)

# =========================
# 完了
# =========================
print("\nSaved results.")

print("Done.")
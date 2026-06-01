import pandas as pd
import numpy as np
from tqdm import tqdm

# =========================
# ファイル
# =========================
PRICE_FILE = "stock_data/prices.parquet"
SHARES_FILE = "stock_data/shares_outstanding.parquet"

OUTPUT_SCORE = "stock_data/accumulation_score.parquet"
OUTPUT_RANKING = "stock_data/accumulation_ranking.csv"

# =========================
# フィルター条件
# =========================
MIN_MARKET_CAP = 50e8      # 50億円
MAX_MARKET_CAP = 5000e8    # 5000億円

MIN_TRADING_VALUE = 1e8    # 1億円

# =========================
# 読み込み
# =========================
print("Loading prices...")

df_price = pd.read_parquet(PRICE_FILE)

df_price["Date"] = pd.to_datetime(
    df_price["Date"]
)

df_shares = pd.read_parquet(
    SHARES_FILE
)

# =========================
# 結合
# =========================
df = df_price.merge(
    df_shares[
        [
            "Ticker",
            "SharesOutstanding"
        ]
    ],
    on="Ticker",
    how="left"
)

df = df.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)

# =========================
# ATR
# =========================
def calc_atr(df_t):

    tr1 = (
        df_t["High"]
        - df_t["Low"]
    )

    tr2 = (
        df_t["High"]
        - df_t["Close"].shift(1)
    ).abs()

    tr3 = (
        df_t["Low"]
        - df_t["Close"].shift(1)
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return tr

# =========================
# 銘柄別
# =========================
results = []

for ticker, df_t in tqdm(
    df.groupby("Ticker"),
    total=df["Ticker"].nunique()
):

    df_t = df_t.copy()

    if len(df_t) < 252:
        continue

    close = df_t["Close"]
    high = df_t["High"]
    low = df_t["Low"]
    volume = df_t["Volume"]

    # =====================
    # 売買代金
    # =====================
    trading_value = (
        close * volume
    )

    avg_value20 = (
        trading_value
        .rolling(20)
        .mean()
    )

    # =====================
    # 時価総額
    # =====================
    market_cap = (
        close
        * df_t["SharesOutstanding"]
    )

    # =====================
    # 出来高倍率
    # =====================
    vol20 = (
        volume
        .rolling(20)
        .mean()
    )

    vol120 = (
        volume
        .rolling(120)
        .mean()
    )

    vol_ratio = (
        vol20 / vol120
    )

    # =====================
    # OBV
    # =====================
    obv = (
        np.sign(
            close.diff()
        ).fillna(0)
        * volume
    ).cumsum()

    obv_growth = (
        obv
        / obv.shift(60)
        - 1
    )

    # =====================
    # 52週高値接近率
    # =====================
    high252 = (
        high
        .rolling(252)
        .max()
    )

    high_ratio = (
        close / high252
    )

    # =====================
    # 高値圏滞在率
    # =====================
    ma25 = (
        close
        .rolling(25)
        .mean()
    )

    stay_rate = (
        (close > ma25)
        .rolling(60)
        .mean()
    )

    # =====================
    # ATR収縮
    # =====================
    tr = calc_atr(df_t)

    atr20 = (
        tr
        .rolling(20)
        .mean()
    )

    atr120 = (
        tr
        .rolling(120)
        .mean()
    )

    atr_ratio = (
        atr20 / atr120
    )

    # =====================
    # 異常吸収指数
    # =====================
    future_ret = (
        close.shift(-5)
        / close
        - 1
    )

    absorption = (
        vol_ratio
        / (
            future_ret.abs()
            + 0.01
        )
    )

    # =====================
    # 保存
    # =====================
    df_t["MarketCap"] = market_cap
    df_t["AvgTradingValue20"] = avg_value20

    df_t["VolRatio"] = vol_ratio
    df_t["OBVGrowth"] = obv_growth
    df_t["HighRatio"] = high_ratio
    df_t["StayRate"] = stay_rate
    df_t["ATRRatio"] = atr_ratio
    df_t["Absorption"] = absorption

    results.append(df_t)

# =========================
# 結合
# =========================
df_score = pd.concat(
    results,
    ignore_index=True
)

# =========================
# 全日付スコア計算
# =========================
print("Calculating scores...")

score_frames = []

for date, daily in tqdm(
    df_score.groupby("Date"),
    total=df_score["Date"].nunique()
):

    daily = daily.copy()

    # ---------------------
    # フィルター
    # ---------------------
    daily = daily[
        daily["MarketCap"]
        >= MIN_MARKET_CAP
    ]

    daily = daily[
        daily["MarketCap"]
        <= MAX_MARKET_CAP
    ]

    daily = daily[
        daily["AvgTradingValue20"]
        >= MIN_TRADING_VALUE
    ]

    if len(daily) < 30:
        continue

    # ---------------------
    # Z-score化
    # ---------------------
    factors = [
        "VolRatio",
        "OBVGrowth",
        "HighRatio",
        "StayRate",
        "Absorption"
    ]

    for col in factors:

        mean = daily[col].mean()
        std = daily[col].std()

        if pd.isna(std) or std == 0:

            daily[col + "_Z"] = 0

        else:

            daily[col + "_Z"] = (
                daily[col] - mean
            ) / std

    atr_mean = (
        daily["ATRRatio"]
        .mean()
    )

    atr_std = (
        daily["ATRRatio"]
        .std()
    )

    if pd.isna(atr_std) or atr_std == 0:

        daily["ATR_Z"] = 0

    else:

        daily["ATR_Z"] = -(
            daily["ATRRatio"]
            - atr_mean
        ) / atr_std

    # ---------------------
    # 最終スコア
    # ---------------------
    daily["AccumulationScore"] = (
          daily["VolRatio_Z"] * 0.25
        + daily["OBVGrowth_Z"] * 0.25
        + daily["HighRatio_Z"] * 0.20
        + daily["StayRate_Z"] * 0.15
        + daily["Absorption_Z"] * 0.10
        + daily["ATR_Z"] * 0.05
    )

    # ---------------------
    # 業種内順位
    # ---------------------
    daily["IndustryRankPct"] = (
        daily.groupby("Industry")
        ["AccumulationScore"]
        .rank(pct=True)
    )

    score_frames.append(daily)

# =========================
# 全期間スコア
# =========================
df_score = pd.concat(
    score_frames,
    ignore_index=True
)

# =========================
# 最新ランキング
# =========================
latest_date = (
    df_score["Date"]
    .max()
)

latest = (
    df_score[
        df_score["Date"]
        == latest_date
    ]
    .sort_values(
        "AccumulationScore",
        ascending=False
    )
)

# =========================
# 保存
# =========================
df_score.to_parquet(
    OUTPUT_SCORE,
    index=False
)

latest.to_csv(
    OUTPUT_RANKING,
    index=False,
    encoding="utf-8-sig"
)

# =========================
# TOP30
# =========================
print("\n=== TOP30 ===")

print(
    latest[
        [
            "Ticker",
            "Name",
            "Industry",
            "MarketCap",
            "AccumulationScore",
            "IndustryRankPct"
        ]
    ]
    .head(30)
)

print("\nSaved")

print(OUTPUT_SCORE)
print(OUTPUT_RANKING)
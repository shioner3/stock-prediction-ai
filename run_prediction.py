import pandas as pd
import numpy as np
import os
import pickle
from lightgbm import LGBMRegressor
from datetime import datetime
from pandas.tseries.offsets import BDay


# =========================
# 設定
# =========================
TOP_N = 5
WEAK_TOP_PERCENT = 0.01

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

FREE_CSV_PATH = "today_picks_free.csv"
PREMIUM_CSV_PATH = os.path.join(LOG_DIR, "today_picks_premium.csv")
PRED_LOG_PATH = os.path.join(LOG_DIR, "predictions.csv")


FEATURES = [
    "Return_1_rank",
    "MA5_ratio_rank",
    "MA25_ratio_rank",
    "MA75_ratio_rank",
    "Volatility_rank",
    "Volume_change_rank",
    "HL_range_rank",
    "RSI_rank"
]

TARGET = "Target"


# =========================
# 🔥 列名吸収（最小追加）
# =========================
def normalize_columns(df):
    df = df.copy()

    rename_map = {}

    if "コード" not in df.columns and "Ticker" in df.columns:
        rename_map["Ticker"] = "コード"

    if "銘柄名" not in df.columns and "Name" in df.columns:
        rename_map["Name"] = "銘柄名"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


# =========================
# ランク正規化
# =========================
def normalize(df):
    df = df.copy()
    if "Pred" not in df.columns:
        raise KeyError("Pred column missing")

    df["PredRank"] = df["Pred"].rank(ascending=False, method="first")
    df = df.sort_values("PredRank")
    df["PredRank"] = range(1, len(df) + 1)
    return df


# =========================
# レジーム判定
# =========================
def get_regime(score):
    if score > 0.55:
        return "strong"
    elif score > 0.52:
        return "slightly_strong"
    elif score > 0.5:
        return "neutral"
    else:
        return "weak"


# =========================
# 戦略
# =========================
def generate_global_strategy():
    return """
========================
■ AI戦略の前提
========================
■ エントリー：即時
■ 保有：5営業日
■ 利確：5営業日後
■ 損切り：-3%
"""


# =========================
# 日次判断
# =========================
def generate_daily_decision(full_df):

    market_score = full_df["Pred"].mean()
    regime = get_regime(market_score)

    best_n = 3
    weak_picks = pd.DataFrame()

    if regime == "weak":

        threshold = full_df["Pred"].quantile(1 - WEAK_TOP_PERCENT)

        weak_picks = full_df[
            (full_df["Pred"] >= threshold) &
            (full_df["Volume_change_rank"] <= 10)
        ].copy()

        weak_picks = normalize(weak_picks)

    if regime == "strong":
        trend = "強気"
        action = "積極エントリー"
    elif regime == "slightly_strong":
        trend = "やや強気"
        action = "選別エントリー"
    elif regime == "neutral":
        trend = "中立"
        action = "様子見"
    else:
        trend = "弱気"
        action = "見送り（例外あり）"

    text = f"""
========================
■ 本日のAI判断
========================
市場評価: {trend}
行動: {action}
"""

    return text, regime, best_n, weak_picks


# =========================
# 記事生成
# =========================
def generate_article(premium_df, daily_comment):

    texts = [daily_comment, generate_global_strategy()]

    selected = premium_df.copy()
    selected = selected.head(TOP_N)
    selected = normalize(selected)

    for _, row in selected.iterrows():
        texts.append(f"""
========================
{row["PredRank"]}位: {row.get("銘柄名","-")} ({row.get("コード","-")})
========================
Pred: {row["Pred"]:.4f}
Regime: {row.get("regime","-")}
""")

    return "\n".join(texts)


# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df = normalize_columns(df)   # 🔥追加
df["Date"] = pd.to_datetime(df["Date"])
latest_date = df["Date"].max()


# =========================
# モデル
# =========================
retrain = (latest_date.weekday() == 0) or (not os.path.exists(MODEL_PATH))

if retrain:
    train = df[df["Date"] < latest_date]
    model = LGBMRegressor()
    model.fit(train[FEATURES], train[TARGET])

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
else:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)


# =========================
# 今日データ
# =========================
today = df[df["Date"] == latest_date].copy()
today["Pred"] = model.predict(today[FEATURES])
today = normalize(today)


# =========================
# 日次判断
# =========================
daily_comment, regime, best_n, weak_picks = generate_daily_decision(today)


# =========================
# FREE CSV
# =========================
free_csv = today.head(TOP_N)[["コード", "銘柄名", "PredRank"]].copy()
free_csv = free_csv.rename(columns={"PredRank": "順位"})
free_csv.to_csv(FREE_CSV_PATH, index=False)


# =========================
# PREMIUM CSV
# =========================
premium_df = today.copy()

premium_df["regime"] = regime
premium_df["predict_date"] = datetime.now().strftime("%Y-%m-%d")
premium_df["target_date"] = (datetime.now() + BDay(5)).strftime("%Y-%m-%d")

premium_df.to_csv(PREMIUM_CSV_PATH, index=False)


# =========================
# ARTICLE
# =========================
article = generate_article(premium_df, daily_comment)

with open("note_article.txt", "w", encoding="utf-8") as f:
    f.write(article)


# =========================
# LOG
# =========================
log_df = premium_df[
    ["コード", "銘柄名", "Pred", "PredRank", "regime", "predict_date", "target_date"]
].copy()

if os.path.exists(PRED_LOG_PATH):
    old = pd.read_csv(PRED_LOG_PATH)
    log_df = pd.concat([old, log_df], ignore_index=True)

log_df.to_csv(PRED_LOG_PATH, index=False)

print("✅ 完全処理完了（FREE / PREMIUM / ARTICLE 分離完了）")
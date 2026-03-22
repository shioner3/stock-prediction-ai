import pandas as pd
import numpy as np
import os
import pickle
from lightgbm import LGBMRegressor
from datetime import datetime
from pandas.tseries.offsets import BDay


TOP_N = 5
WEAK_TOP_PERCENT = 0.01

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
PERF_LOG = os.path.join(BASE_DIR, "logs/performance.csv")


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
# 共通ユーティリティ（⭐追加）
# =========================
def normalize(df):
    df = df.copy()
    df["PredRank"] = df["Pred"].rank(ascending=False, method="first")
    df = df.sort_values("PredRank")
    df["PredRank"] = range(1, len(df) + 1)
    return df


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
        pos = "80〜100%"
    elif regime == "slightly_strong":
        trend = "やや強気"
        action = "選別エントリー"
        pos = "50〜70%"
    elif regime == "neutral":
        trend = "中立"
        action = "様子見"
        pos = "30%"
    else:
        trend = "弱気"
        action = "見送り（例外あり）"
        pos = "0%"

    text = f"""
========================
■ 本日のAI判断
========================

市場評価: {trend}
行動: {action}
"""

    return text, regime, best_n, weak_picks


# =========================
def generate_article(df, daily_comment, best_n, weak_picks, regime):

    texts = [daily_comment, generate_global_strategy()]

    # =========================
    # ★ここが修正ポイント
    # =========================
    if regime == "weak" and len(weak_picks) > 0:
        selected = weak_picks.head(best_n).copy()
    else:
        selected = df.head(best_n).copy()

    # ⭐必ずrank付け（ここで完全保証）
    selected = normalize(selected)

    for _, row in selected.iterrows():
        texts.append(f"""
========================
{row["PredRank"]}位
========================
""")

    return "\n".join(texts)


# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
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
# 今日
# =========================
today = df[df["Date"] == latest_date].copy()

today["Pred"] = model.predict(today[FEATURES])


# =========================
# ⭐ここ重要（Rank廃止してPred基準に統一）
# =========================
today = normalize(today)


# =========================
# picks
# =========================
picks = today.head(TOP_N).copy()


# =========================
# 日次判断
# =========================
daily_comment, regime, best_n, weak_picks = generate_daily_decision(today)


# =========================
# CSV
# =========================
picks.to_csv("today_picks.csv", index=False)


# =========================
# 記事
# =========================
article = generate_article(picks, daily_comment, best_n, weak_picks, regime)

with open("note_article.txt", "w", encoding="utf-8") as f:
    f.write(article)


print("✅ 完全処理完了")
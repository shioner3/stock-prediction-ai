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

# 月分割ログ
month_str = datetime.now().strftime("%Y-%m")
PRED_LOG_PATH = os.path.join(LOG_DIR, f"predictions_{month_str}.csv")

# 🔥 実績ログ（ここ追加）
PERF_LOG_PATH = os.path.join(LOG_DIR, "performance.csv")


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
# 列名吸収
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
# ランク
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
# レジーム
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
# 🔥 実績読み込み（追加）
# =========================
def load_performance():

    if not os.path.exists(PERF_LOG_PATH):
        return None

    df = pd.read_csv(PERF_LOG_PATH)

    if len(df) < 10:
        return None

    # 直近100件
    df = df.tail(100)

    result = {}

    # 全体
    result["all"] = {
        "win_rate": df["win"].mean(),
        "avg_return": df["return"].mean(),
        "sharpe": df["return"].mean() / df["return"].std() if df["return"].std() != 0 else 0
    }

    # レジーム別
    result["regime"] = {}

    for r in ["strong", "slightly_strong", "neutral", "weak"]:
        df_r = df[df["regime"] == r]

        if len(df_r) < 5:
            continue

        result["regime"][r] = {
            "win_rate": df_r["win"].mean(),
            "avg_return": df_r["return"].mean(),
            "sharpe": df_r["return"].mean() / df_r["return"].std() if df_r["return"].std() != 0 else 0
        }

    return result


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
# 記事生成（🔥実績追加）
# =========================
def generate_article(premium_df, daily_comment):

    texts = [daily_comment, generate_global_strategy()]

    # 🔥 実績追加
    perf = load_performance()

    if perf is not None:

        texts.append("\n========================\n■ 実績（直近）\n========================")

        texts.append(f"""
勝率: {perf["all"]["win_rate"]:.2%}
平均リターン: {perf["all"]["avg_return"]:.2%}
Sharpe: {perf["all"]["sharpe"]:.2f}
""")

        texts.append("\n■ レジーム別")

        for r, v in perf["regime"].items():
            texts.append(f"""
[{r}]
勝率: {v["win_rate"]:.2%}
平均リターン: {v["avg_return"]:.2%}
Sharpe: {v["sharpe"]:.2f}
""")

    # 銘柄
    selected = premium_df.head(TOP_N).copy()
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
df = normalize_columns(df)

df["Date"] = pd.to_datetime(df["Date"])
latest_date = df["Date"].max()

print("\n=== PREDICTION DEBUG ===")
print("latest_date used:", latest_date)
print("today rows:", len(df[df["Date"] == latest_date]))
print("========================")


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
# LOG（月分割）
# =========================
log_df = premium_df[
    ["コード", "銘柄名", "Pred", "PredRank", "regime", "predict_date", "target_date"]
].copy()

if os.path.exists(PRED_LOG_PATH):
    old = pd.read_csv(PRED_LOG_PATH)
    log_df = pd.concat([old, log_df], ignore_index=True)

log_df.to_csv(PRED_LOG_PATH, index=False)


print(f"✅ 完了（実績込み記事生成）")


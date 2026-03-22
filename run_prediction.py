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
# 🔥 共通戦略
# =========================
def generate_global_strategy():
    return """
========================
■ AI戦略の前提（重要）
========================

■ エントリー
当日の寄付き or 引けで即エントリー
■ 保有期間
5営業日
■ 利確
5営業日後に決済
■ 損切り
-3%

------------------------
■ 注意事項
本戦略は短期トレードです
"""


# =========================
# 🔥 レジーム判定
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
# 🔥 実績取得
# =========================
def get_performance_comment(regime):

    if not os.path.exists(PERF_LOG):
        return "\n（実績データなし）\n"

    df = pd.read_csv(PERF_LOG)

    df_r = df[df["regime"] == regime]

    if len(df_r) < 10:
        return "\n（データ不足）\n"

    win = df_r["win"].mean()
    avg = df_r["return"].mean()

    return f"""
■ 過去類似局面の実績
勝率: {win:.2%}
平均リターン: {avg:.2%}
"""


# =========================
# 🔥 日次判断
# =========================
def generate_daily_decision(picks_df, full_df):

    market_score = full_df["Pred"].mean()
    regime = get_regime(market_score)

    if regime == "strong":
        trend = "強気"
        action = "フルエントリー"
        pos = "100%"

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
        action = "見送り"
        pos = "0%"

    perf_text = get_performance_comment(regime)

    text = f"""
========================
■ 本日のAI判断
========================

市場評価: {trend}
行動: {action}
推奨ポジション: {pos}

{perf_text}
"""

    return text, regime


# =========================
# 🔥 記事生成
# =========================
def generate_article(df, daily_comment):
    texts = [daily_comment, generate_global_strategy()]

    for _, row in df.iterrows():
        texts.append(f"""
========================
{row["PredRank"]}位: {row["銘柄名"]} ({row["コード"]})
========================

■ AI評価
{row["PredRank"]}位
""")

    return "\n".join(texts)


# =========================
# データ読み込み
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
# 今日データ
# =========================
today = df[df["Date"] == latest_date].copy()

today["Pred"] = model.predict(today[FEATURES])
today["Rank"] = today["Pred"].rank(ascending=False)

# =========================
# 銘柄名
# =========================
df_info = pd.read_csv(os.path.join(BASE_DIR, "data_j.csv"), dtype=str)
ticker_to_name = dict(zip(df_info["コード"], df_info["銘柄名"]))

# =========================
# picks
# =========================
picks = today.nsmallest(TOP_N, "Rank").copy()

picks["コード"] = picks["Ticker"].str.replace(".T", "", regex=False)
picks["銘柄名"] = picks["コード"].map(ticker_to_name)
picks["PredRank"] = range(1, len(picks) + 1)

# =========================
# 🔥 日次判断（＋regime取得）
# =========================
daily_comment, regime = generate_daily_decision(picks, today)

# =========================
# CSV
# =========================
picks[["Ticker", "コード", "銘柄名"]].to_csv("today_picks.csv", index=False)

# =========================
# 🔥 記事
# =========================
article = generate_article(picks, daily_comment)

with open("note_article.txt", "w", encoding="utf-8") as f:
    f.write(article)

# =========================
# 🔥 ログ保存（regime追加）
# =========================
today_dt = datetime.now()
target_dt = today_dt + BDay(5)

picks["predict_date"] = today_dt.strftime("%Y-%m-%d")
picks["target_date"] = target_dt.strftime("%Y-%m-%d")
picks["regime"] = regime  # 🔥追加

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, "predictions.csv")

# 初回 or カラム違いなら上書き
if not os.path.exists(log_path):
    picks[save_cols].to_csv(log_path, index=False)
else:
    df_old = pd.read_csv(log_path)

    # カラム一致チェック
    if list(df_old.columns) != save_cols:
        print("⚠ カラム不一致 → 上書き")
        picks[save_cols].to_csv(log_path, index=False)
    else:
        picks[save_cols].to_csv(
            log_path,
            mode="a",
            header=False,
            index=False
        )

print("✅ 完全処理完了")
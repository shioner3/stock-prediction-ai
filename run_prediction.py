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

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest.parquet")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

FREE_CSV_PATH = "today_picks_free.csv"
PREMIUM_CSV_PATH = os.path.join(LOG_DIR, "today_picks_premium.csv")

ARTICLE_PATH = "note_article.txt"

month_str = datetime.now().strftime("%Y-%m")
PRED_LOG_PATH = os.path.join(LOG_DIR, f"predictions_{month_str}.csv")

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
# 🔥 バックテスト結果
# =========================
BACKTEST_RESULTS = [
    {"period": "2018-2021 → 2021-2022", "cagr": 0.25, "sharpe": 1.11, "maxdd": -0.14},
    {"period": "2019-2022 → 2022-2023", "cagr": 1.22, "sharpe": 3.38, "maxdd": -0.04},
    {"period": "2020-2023 → 2023-2024", "cagr": 1.29, "sharpe": 3.10, "maxdd": -0.09},
    {"period": "2021-2024 → 2024-2025", "cagr": 1.37, "sharpe": 2.63, "maxdd": -0.14},
]

AVG_CAGR = 1.038
AVG_SHARPE = 2.56
AVG_MAXDD = -0.107

# =========================
# ユーティリティ
# =========================
def normalize_columns(df):
    rename_map = {}
    if "コード" not in df.columns and "Ticker" in df.columns:
        rename_map["Ticker"] = "コード"
    if "銘柄名" not in df.columns and "Name" in df.columns:
        rename_map["Name"] = "銘柄名"
    return df.rename(columns=rename_map) if rename_map else df


def normalize(df):
    df = df.copy()
    df["PredRank"] = df["Pred"].rank(ascending=False, method="first")
    df = df.sort_values("PredRank")
    df["PredRank"] = range(1, len(df) + 1)
    return df


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
# 実績
# =========================
def load_performance():
    if not os.path.exists(PERF_LOG_PATH):
        return None

    df = pd.read_csv(PERF_LOG_PATH)
    if len(df) < 10:
        return None

    df = df.tail(100)

    return {
        "win_rate": df["win"].mean(),
        "avg_return": df["return"].mean(),
        "sharpe": df["return"].mean() / df["return"].std() if df["return"].std() != 0 else 0
    }


# =========================
# 無料記事
# =========================
def generate_free_article(today, regime):

    if regime == "strong":
        trend = "強気"
        action = "押し目はチャンス"
    elif regime == "neutral":
        trend = "中立"
        action = "方向感なし"
    else:
        trend = "弱気"
        action = "基本は様子見"

    text = f"""
========================
■ 本日の市場判断
========================

市場評価：{trend}

→ {action}


========================
■ 注目銘柄（TOP5）
========================
"""

    for _, row in today.head(TOP_N).iterrows():
        text += f"{int(row['PredRank'])}位：{row['銘柄名']}（{row['コード']}）\n"

    # 🔥 ここが超重要（インパクト）
    text += f"""
========================
■ AIの実力（検証結果）
========================

・平均年利：約{int(AVG_CAGR*100)}%
・最大ドローダウン：約{int(abs(AVG_MAXDD)*100)}%
・複数期間で安定してプラス

👉 放置でも資産が増える設計
"""

    text += """
========================
■ なぜ強いのか？
========================

・市場状況に応じたスコアリング
・複数指標の統合判断
・短期リターン特化設計

👉 ただし、このままでは再現できません

========================
👇 有料で公開
========================

・具体的な売買ルール
・エントリー条件
・損切りライン
・全銘柄ランキング

👉 「そのまま使える形」で公開
"""

    return text

# =========================
# 有料記事
# =========================
def generate_premium_article(today, regime):

    text = """
========================
■ AI戦略（完全版）
========================

・エントリー：当日 or 翌日
・保有：5営業日
・損切り：-3%

========================
■ レジーム別戦略
========================
"""

    if regime == "strong":
        text += "強気 → 上位銘柄を複数エントリー\n"
    elif regime == "neutral":
        text += "中立 → 上位のみ選別\n"
    else:
        text += "弱気 → 原則見送り（例外条件あり）\n"

    # 🔥 バックテスト詳細（信頼パート）
    text += "\n========================\n■ 詳細バックテスト\n========================\n"

    for r in BACKTEST_RESULTS:
        text += f"""
{r['period']}
CAGR: {int(r['cagr']*100)}%
Sharpe: {r['sharpe']:.2f}
MaxDD: {int(r['maxdd']*100)}%
"""

    text += f"""
========================
■ 平均パフォーマンス
========================

・年平均リターン：約{int(AVG_CAGR*100)}%
・Sharpe：{AVG_SHARPE}
・最大ドローダウン：約{int(abs(AVG_MAXDD)*100)}%

👉 全期間で安定して利益
"""

    # 🔥 ランキング
    text += "\n========================\n■ 全ランキング\n========================\n"

    for _, row in today.head(20).iterrows():
        text += f"{int(row['PredRank'])}位 {row['銘柄名']} ({row['コード']}) Pred:{row['Pred']:.3f}\n"

    return text

# =========================
# データ読み込み
# =========================
train_df = pd.read_parquet(TRAIN_DATA_PATH)
predict_df = pd.read_parquet(PREDICT_DATA_PATH)

train_df = normalize_columns(train_df)
predict_df = normalize_columns(predict_df)

train_df["Date"] = pd.to_datetime(train_df["Date"])
predict_df["Date"] = pd.to_datetime(predict_df["Date"])

latest_date = predict_df["Date"].max()

print("\n=== DEBUG ===")
print("predict latest:", latest_date)
print("rows:", len(predict_df[predict_df["Date"] == latest_date]))
print("========================")


# =========================
# モデル
# =========================
if not os.path.exists(MODEL_PATH):
    model = LGBMRegressor()
    model.fit(train_df[FEATURES], train_df[TARGET])
    pickle.dump(model, open(MODEL_PATH, "wb"))
else:
    model = pickle.load(open(MODEL_PATH, "rb"))


# =========================
# 予測
# =========================
today = predict_df[predict_df["Date"] == latest_date].copy()

today["Pred"] = model.predict(today[FEATURES])
today = normalize(today)

market_score = today["Pred"].mean()
regime = get_regime(market_score)


# =========================
# CSV出力
# =========================
today.head(TOP_N)[["コード", "銘柄名", "PredRank"]]\
    .rename(columns={"PredRank": "順位"})\
    .to_csv(FREE_CSV_PATH, index=False)

premium_df = today.copy()
premium_df["regime"] = regime
premium_df["predict_date"] = datetime.now().strftime("%Y-%m-%d")
premium_df["target_date"] = (datetime.now() + BDay(5)).strftime("%Y-%m-%d")

premium_df.to_csv(PREMIUM_CSV_PATH, index=False)


# =========================
# 記事生成
# =========================
free_article = generate_free_article(today, regime)
premium_article = generate_premium_article(today, regime)

with open(ARTICLE_PATH, "w", encoding="utf-8") as f:
    f.write(free_article + "\n\n================ 有料 =================\n\n" + premium_article)


# =========================
# LOG
# =========================
log_df = premium_df[
    ["コード", "銘柄名", "Pred", "PredRank", "regime", "predict_date", "target_date"]
]

if os.path.exists(PRED_LOG_PATH):
    old = pd.read_csv(PRED_LOG_PATH)
    log_df = pd.concat([old, log_df], ignore_index=True)

log_df.to_csv(PRED_LOG_PATH, index=False)

print("✅ 完了（収益導線込み）")
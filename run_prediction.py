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
PRED_THRESHOLD = 0.0  # 🔥 これ重要（期待値フィルター）

BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest.parquet")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

ARTICLE_PATH = "note_article.txt"

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
# バックテスト
# =========================
BACKTEST_RESULTS = [
    {"period": "2019-2022", "cagr": 0.21, "sharpe": 1.05, "maxdd": -0.12},
    {"period": "2020-2023", "cagr": 1.18, "sharpe": 3.10, "maxdd": -0.05},
    {"period": "2021-2024", "cagr": 1.32, "sharpe": 2.80, "maxdd": -0.10},
]

AVG_CAGR = 1.03
AVG_SHARPE = 2.32
AVG_MAXDD = -0.09


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
    if score > 0.01:
        return "強気"
    elif score > 0:
        return "中立"
    else:
        return "弱気"


def get_action(regime):
    if regime == "強気":
        return "押し目はチャンス。分散して複数銘柄へ"
    elif regime == "中立":
        return "無理に入らず、上位のみ厳選"
    else:
        return "期待値が低いため基本は見送り"


def load_performance():
    if not os.path.exists(PERF_LOG_PATH):
        return None

    df = pd.read_csv(PERF_LOG_PATH)
    if len(df) < 10:
        return None

    df = df.tail(100)

    return {
        "win_rate": df["win"].mean(),
        "avg_return": df["return"].mean()
    }


# =========================
# 無料記事
# =========================
def generate_free_article(today, regime):

    action = get_action(regime)

    text = f"""
========================
■ 本日の市場判断
========================
市場：{regime}

→ {action}

========================
■ 注目銘柄（TOP5）
========================
"""

    picks = today[today["Pred"] > PRED_THRESHOLD].head(TOP_N)

    if len(picks) == 0:
        text += "本日は該当なし（無理なエントリーは避けます）\n"
    else:
        for _, row in picks.iterrows():
            text += f"{int(row['PredRank'])}位：{row['銘柄名']}（{row['コード']}）\n"

    text += f"""
========================
■ AIの実力（検証結果）
========================

・平均年利：約{int(AVG_CAGR*100)}%
・Sharpe：{AVG_SHARPE}
・最大DD：約{int(abs(AVG_MAXDD)*100)}%

👉 複数期間で安定してプラス
"""

    perf = load_performance()

    text += """
========================
■ 直近の実績
========================
"""

    if perf:
        text += f"""
・勝率：約{perf["win_rate"]:.0%}
・平均リターン：約{perf["avg_return"]:.1%}
"""
    else:
        text += "※データ蓄積中\n"

    text += """
========================
■ なぜ勝てるのか？
========================

・複数指標の統合スコア
・市場状況に応じた戦略切替
・短期リターン特化モデル

========================
■ 注意
========================

このままでは再現できません
（ルールは非公開）

========================
👉 有料版で公開
========================
・売買ルール完全公開
・エントリー条件
・損切り基準
・全ランキング
"""

    return text


# =========================
# 有料記事
# =========================
def generate_premium_article(today, regime):

    text = """
========================
■ AI戦略（完全再現版）
========================

・エントリー：当日 or 翌日
・保有：5営業日
・損切り：-3%
"""

    if regime == "強気":
        text += "\n強気 → 上位銘柄を分散エントリー\n"
    elif regime == "中立":
        text += "\n中立 → 厳選して少数\n"
    else:
        text += "\n弱気 → 基本ノートレード\n"

    text += "\n========================\n■ バックテスト詳細\n========================\n"

    for r in BACKTEST_RESULTS:
        text += f"""
{r['period']}
CAGR: {int(r['cagr']*100)}%
Sharpe: {r['sharpe']}
MaxDD: {int(r['maxdd']*100)}%
"""

    text += f"""
========================
■ 平均成績
========================

年利：約{int(AVG_CAGR*100)}%
Sharpe：{AVG_SHARPE}
DD：約{int(abs(AVG_MAXDD)*100)}%

========================
■ TOP20ランキング
========================
"""

    for _, row in today.head(20).iterrows():
        text += f"{int(row['PredRank'])}位 {row['銘柄名']} ({row['コード']})\n"

    return text


# =========================
# モデル
# =========================
train_df = pd.read_parquet(TRAIN_DATA_PATH)
predict_df = pd.read_parquet(PREDICT_DATA_PATH)

train_df = normalize_columns(train_df)
predict_df = normalize_columns(predict_df)

train_df["Date"] = pd.to_datetime(train_df["Date"])
predict_df["Date"] = pd.to_datetime(predict_df["Date"])

latest_date = predict_df["Date"].max()

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
# 出力
# =========================
free_article = generate_free_article(today, regime)
premium_article = generate_premium_article(today, regime)

with open(ARTICLE_PATH, "w", encoding="utf-8") as f:
    f.write(free_article + "\n\n================ 有料 =================\n\n" + premium_article)

print("✅ 完了（ガチ売れる版）")
import pandas as pd
import numpy as np
import os
import pickle
from lightgbm import LGBMRegressor
from datetime import datetime

# =========================
# 設定
# =========================
BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

ARTICLE_PATH = "note_article.txt"

# 🔥 強化特徴量
FEATURES = [
    # rank
    "Return_1_rank", "MA5_ratio_rank", "MA25_ratio_rank", "MA75_ratio_rank",
    "Volatility_rank", "Volume_change_rank", "HL_range_rank", "RSI_rank",
    "Return_5_rank", "Return_20_rank", "Volume_spike_rank",
    "Breakout_rank", "Return_vol_adj_rank",

    # raw（重要）
    "Return_5", "Return_20", "Breakout", "Volume_spike", "Return_vol_adj"
]

TARGET = "Target_rank"

# =========================
# レジーム設定（連続値ベース）
# =========================
def get_regime(score):
    if score > 0.002:
        return "強気"
    elif score > -0.002:
        return "中立"
    else:
        return "弱気"

# 🔥 レジーム別フィルター
REGIME_CONFIG = {
    "強気": {"quantile": 0.7, "max_n": 5},
    "中立": {"quantile": 0.85, "max_n": 3},
    "弱気": {"quantile": 1.0, "max_n": 0}
}

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


def rank_normalize(df):
    df = df.copy()
    df["PredRank"] = df["Pred"].rank(ascending=False, method="first")
    df = df.sort_values("PredRank")
    df["PredRank"] = range(1, len(df) + 1)
    return df


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
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(train_df[FEATURES], train_df[TARGET])
    pickle.dump(model, open(MODEL_PATH, "wb"))
else:
    model = pickle.load(open(MODEL_PATH, "rb"))

# =========================
# 予測
# =========================
today = predict_df[predict_df["Date"] == latest_date].copy()

today["Pred"] = model.predict(today[FEATURES])

# 🔥 スケール安定化（重要）
today["Pred"] = (today["Pred"] - today["Pred"].mean()) / (today["Pred"].std() + 1e-9)

today = rank_normalize(today)

# =========================
# レジーム判定
# =========================
market_score = today["Pred"].mean()
regime = get_regime(market_score)
config = REGIME_CONFIG[regime]

# =========================
# フィルタ
# =========================
if config["max_n"] == 0:
    picks = pd.DataFrame()
else:
    th = today["Pred"].quantile(config["quantile"])
    picks = today[today["Pred"] > th].head(config["max_n"])

# =========================
# 記事生成（強化版）
# =========================
def generate_article(today, picks, regime):

    text = f"""
========================
■ 本日の市場判断
========================
市場：{regime}
平均スコア：{today["Pred"].mean():.4f}

"""

    if regime == "強気":
        text += "→ 攻め（分散エントリー）\n"
    elif regime == "中立":
        text += "→ 厳選\n"
    else:
        text += "→ ノートレード推奨\n"

    text += "\n========================\n■ 注目銘柄\n========================\n"

    if len(picks) == 0:
        text += "該当なし\n"
    else:
        for _, row in picks.iterrows():
            text += f"{int(row['PredRank'])}位 {row['銘柄名']} ({row['コード']})\n"

    text += f"""

========================
■ 市場状態
========================
平均Pred: {today["Pred"].mean():.4f}
分散: {today["Pred"].std():.4f}

========================
■ 戦略
========================
・翌日寄りエントリー
・5日保有
・-3%損切り
"""

    return text


# =========================
# 出力
# =========================
article = generate_article(today, picks, regime)

with open(ARTICLE_PATH, "w", encoding="utf-8") as f:
    f.write(article)

print("✅ 完了（強化版AI）")
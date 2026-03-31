import pandas as pd
import numpy as np
import os
import pickle
from lightgbm import LGBMClassifier
from datetime import datetime

# =========================
# 設定
# =========================
TOP_N = 5
HOLD_DAYS = 5  # 🔥 featureと統一

BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest.parquet")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
MODEL_META_PATH = os.path.join(BASE_DIR, "model_meta.pkl")

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

ARTICLE_PATH = "note_article.txt"

month_str = datetime.now().strftime("%Y-%m")
PRED_LOG_PATH = os.path.join(LOG_DIR, f"predictions_{month_str}.csv")

PERF_LOG_PATH = os.path.join(LOG_DIR, "performance.csv")

# =========================
# 特徴量（🔥 生値に変更）
# =========================
FEATURES = [
    "Return_1","Return_3",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio",
    "HL_range","RSI"
]

TARGET = "Target"

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
    df = df.sort_values("Pred", ascending=False)
    df["PredRank"] = range(1, len(df) + 1)
    return df


def get_regime(score):
    if score > 0.56:
        return "strong"
    elif score > 0.53:
        return "slightly_strong"
    elif score > 0.51:
        return "neutral"
    else:
        return "weak"


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
current_month = latest_date.strftime("%Y-%m")

# =========================
# 🔥 Targetチェック（追加）
# =========================
print("\n=== TARGET CHECK ===")
print("Target mean:", train_df["Target"].mean())

# =========================
# 月次学習判定
# =========================
retrain = True

if os.path.exists(MODEL_PATH) and os.path.exists(MODEL_META_PATH):
    meta = pickle.load(open(MODEL_META_PATH, "rb"))
    last_train_month = meta.get("train_month")

    if last_train_month == current_month:
        retrain = False

# =========================
# モデル
# =========================
if retrain:
    print("🔄 モデル再学習")

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(train_df[FEATURES], train_df[TARGET])

    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump({"train_month": current_month}, open(MODEL_META_PATH, "wb"))

else:
    print("⚡ 既存モデル使用")
    model = pickle.load(open(MODEL_PATH, "rb"))

# =========================
# 予測
# =========================
today = predict_df[predict_df["Date"] == latest_date].copy()

today["Pred"] = model.predict_proba(today[FEATURES])[:, 1]

# =========================
# 🔥 予測分布チェック（追加）
# =========================
print("\n=== PRED CHECK ===")
print(today["Pred"].describe())

# =========================
# フィルター
# =========================
if "limit_up_flag" in today.columns:
    today = today[today["limit_up_flag"] == 0]

if "Volume" in today.columns:
    today = today[today["Volume"].fillna(0) > 10000]

# =========================
# 🔥 Top Nだけ使う（シンプル）
# =========================
today = today.sort_values("Pred", ascending=False).head(TOP_N)

if len(today) == 0:
    print("⚠️ 銘柄なし")
    exit()

# =========================
# ランキング
# =========================
today = normalize(today)

# =========================
# レジーム
# =========================
market_score = today["Pred"].mean()
regime = get_regime(market_score)

# =========================
# ログ保存
# =========================
today["predict_date"] = latest_date
today["target_date"] = latest_date + pd.Timedelta(days=HOLD_DAYS)

if os.path.exists(PRED_LOG_PATH):
    today.to_csv(PRED_LOG_PATH, mode="a", header=False, index=False)
else:
    today.to_csv(PRED_LOG_PATH, index=False)

# =========================
# 出力
# =========================
print("\n=== 今日の銘柄 ===")
print(today[["コード", "銘柄名", "Pred"]])

print("\n市場状態:", regime)
print("銘柄数:", len(today))
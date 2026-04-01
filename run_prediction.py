import pandas as pd
import numpy as np
import os
import pickle
from lightgbm import LGBMClassifier
from datetime import datetime

# =========================
# 設定
# =========================
TOP_N = 3
HOLD_DAYS = 3

CANDIDATES = 20
THRESHOLD = 0.32

BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest.parquet")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
MODEL_META_PATH = os.path.join(BASE_DIR, "model_meta.pkl")

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

month_str = datetime.now().strftime("%Y-%m")
PRED_LOG_PATH = os.path.join(LOG_DIR, f"predictions_{month_str}.csv")

# =========================
# FEATURES（backtestと完全一致）
# =========================
FEATURES = [
    "Return_1","Return_3","Return_5",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio","Volume_accel",
    "HL_range",
    "EMA_gap",
    "Momentum_5","Momentum_10",
    "ATR_ratio",
    "RSI"
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
# TARGETチェック
# =========================
print("\n=== TARGET CHECK ===")
print("Target mean:", train_df["Target"].mean())

# =========================
# モデル再学習判定
# =========================
retrain = True

if os.path.exists(MODEL_PATH) and os.path.exists(MODEL_META_PATH):
    meta = pickle.load(open(MODEL_META_PATH, "rb"))
    last_train_month = meta.get("train_month")

    if last_train_month == current_month:
        retrain = False

# =========================
# 学習
# =========================
if retrain:
    print("🔄 モデル再学習")

    train_df[FEATURES] = train_df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

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
# 予測データ
# =========================
today = predict_df[predict_df["Date"] == latest_date].copy()

# =========================
# feature安全化
# =========================
today[FEATURES] = today[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

missing_cols = [c for c in FEATURES if c not in today.columns]
if len(missing_cols) > 0:
    raise ValueError(f"Missing features: {missing_cols}")

# =========================
# 予測
# =========================
today["Pred"] = model.predict_proba(today[FEATURES])[:, 1]

# =========================
# 予測分布チェック
# =========================
print("\n=== PRED CHECK ===")
print(today["Pred"].describe())

# =========================
# 基本フィルター
# =========================
if "limit_up_flag" in today.columns:
    today = today[today["limit_up_flag"] == 0]

if "Volume" in today.columns:
    today = today[today["Volume"].fillna(0) > 10000]

today = today.dropna(subset=["Pred"])

if today.empty:
    print("⚠️ 銘柄なし")
    exit()

# =========================
# 🔥 ハイブリッド選定（最重要）
# =========================

# ① 候補（rank）
candidates = today.sort_values("Pred", ascending=False).head(CANDIDATES)

# ② 生値フィルター
filtered = candidates[candidates["Pred"] > THRESHOLD]

# ③ fallback
if len(filtered) < TOP_N:
    filtered = candidates.head(TOP_N)

# ④ 最終
today = filtered.head(TOP_N).copy()

# =========================
# ランキング
# =========================
today["PredRank"] = range(1, len(today) + 1)

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
print(today[["コード", "銘柄名", "Pred", "PredRank"]])

print("\n市場状態:", regime)
print("銘柄数:", len(today))
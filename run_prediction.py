import pandas as pd
import numpy as np
import os
import pickle
from lightgbm import LGBMClassifier
from datetime import datetime

# =========================
# 設定
# =========================
TOP_N = 1
HOLD_DAYS = 7
THRESHOLD = 0.55   # ← Target変更に合わせて上げる
CANDIDATES = 20

BASE_DIR = os.path.dirname(__file__)

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset.parquet")
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "ml_dataset_latest.parquet")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
MODEL_META_PATH = os.path.join(BASE_DIR, "model_meta.pkl")

LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

month_str = datetime.now().strftime("%Y-%m")
PRED_LOG_PATH = os.path.join(LOG_DIR, f"predictions_{month_str}.csv")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "today_picks.csv")

# =========================
# FEATURES（完全一致）
# =========================
FEATURES = [
    "Return_1","Return_3",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio",
    "HL_range",
    "Rel_Return_1"
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
    if score > 0.60:
        return "strong"
    elif score > 0.55:
        return "slightly_strong"
    elif score > 0.50:
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

print(f"\n📅 予測日: {latest_date}")

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
        learning_rate=0.03,
        max_depth=6,
        random_state=42
    )

    model.fit(train_df[FEATURES], train_df[TARGET])

    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump({"train_month": current_month}, open(MODEL_META_PATH, "wb"))

else:
    print("⚡ 既存モデル使用")
    model = pickle.load(open(MODEL_PATH, "rb"))

# =========================
# 今日データ
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

print("\n=== PRED CHECK ===")
print(today["Pred"].describe())

# =========================
# フィルター
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
# 銘柄選定
# =========================
candidates = today.sort_values("Pred", ascending=False).head(CANDIDATES)

filtered = candidates[candidates["Pred"] > THRESHOLD]

if len(filtered) < TOP_N:
    filtered = candidates.head(TOP_N)

today = filtered.head(TOP_N).copy()

# =========================
# ランク
# =========================
today["PredRank"] = range(1, len(today) + 1)

# =========================
# レジーム
# =========================
market_score = today["Pred"].mean()
regime = get_regime(market_score)

# =========================
# ログ
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
today.to_csv(OUTPUT_PATH, index=False)

print("\n=== 今日の銘柄 ===")
print(today[["コード", "銘柄名", "Pred", "PredRank"]])

print("\n市場状態:", regime)
print("銘柄数:", len(today))

print(f"\n📤 出力完了: {OUTPUT_PATH}")
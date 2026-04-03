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
THRESHOLD = 0.55
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
# FEATURES
# =========================
FEATURES = [
    "Return_1","Return_3",
    "MA3_ratio","MA5_ratio","MA10_ratio",
    "Volatility",
    "Volume_change","Volume_ratio",
    "HL_range",
    "Rel_Return_1",
    "Trend_5",
    "Trend_10",
    "Vol_Ratio"
]

TARGET = "Target"

# =========================
# 安全ユーティリティ（ここが重要）
# =========================
def normalize_columns(df):
    if "Ticker" not in df.columns and "コード" in df.columns:
        df = df.rename(columns={"コード": "Ticker"})
    if "Name" not in df.columns and "銘柄名" in df.columns:
        df = df.rename(columns={"銘柄名": "Name"})
    return df

def safe_add_features(df):

    # 念のためTickerがない場合は終了回避
    if "Ticker" not in df.columns:
        df["Ticker"] = "UNKNOWN"

    # Trend系（同日データだと意味薄いがエラー防止）
    df["Trend_5"] = 0
    df["Trend_10"] = 0

    # Vol系
    if "Volatility" not in df.columns:
        df["Volatility"] = 0

    df["Market_Vol"] = df["Volatility"].mean()
    df["Vol_Ratio"] = df["Volatility"] / (df["Market_Vol"] + 1e-9)

    return df

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
# モデル
# =========================
retrain = True

if os.path.exists(MODEL_PATH) and os.path.exists(MODEL_META_PATH):
    meta = pickle.load(open(MODEL_META_PATH, "rb"))
    if meta.get("train_month") == current_month:
        retrain = False

if retrain:
    print("🔄 モデル再学習")

    train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)

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
# 🔥 安全特徴量追加
# =========================
today = safe_add_features(today)

# =========================
# featureチェック（ここが重要）
# =========================
missing_cols = [c for c in FEATURES if c not in today.columns]
if missing_cols:
    raise ValueError(f"Missing features: {missing_cols}")

today[FEATURES] = today[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

# =========================
# 予測
# =========================
today["Pred"] = model.predict_proba(today[FEATURES])[:, 1]

print("\n=== PRED CHECK ===")
print(today["Pred"].describe())

# =========================
# フィルター（安全化）
# =========================
if "limit_up_flag" in today.columns:
    today = today[today["limit_up_flag"] == 0]

if "Volume" in today.columns:
    today = today[today["Volume"].fillna(0) > 10000]

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
# 出力
# =========================
today["PredRank"] = range(1, len(today) + 1)

today["predict_date"] = latest_date
today["target_date"] = latest_date + pd.Timedelta(days=HOLD_DAYS)

today.to_csv(OUTPUT_PATH, index=False)

print("\n=== 今日の銘柄 ===")
print(today[["Ticker", "Name", "Pred", "PredRank"]])

print(f"\n📤 出力完了: {OUTPUT_PATH}")
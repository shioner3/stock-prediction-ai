import subprocess
import os
import sys
import time

# =========================
# 設定
# =========================
PARQUET_FILE = "stock_data/prices.parquet"
TRAIN_FILE = "ml_dataset.parquet"
PREDICT_FILE = "ml_dataset_latest.parquet"

# =========================
# 共通関数
# =========================
def run_script(script_name):
    print(f"\n🚀 Running: {script_name}")
    start = time.time()

    script_path = os.path.join(os.getcwd(), script_name)

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"{script_name} が見つかりません")

    try:
        subprocess.run(
            [sys.executable, script_path],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {script_name}")
        raise e

    elapsed = time.time() - start
    print(f"✅ Finished: {script_name} ({elapsed:.2f}s)")


def check_file_exists(file_path, name):
    if not os.path.exists(file_path):
        raise Exception(f"{name} が生成されていません: {file_path}")


def check_file_size(file_path, min_size, name):
    size = os.path.getsize(file_path)
    print(f"{name} size: {size}")

    if size < min_size:
        raise Exception(f"{name} が小さすぎます（破損の可能性）")


# =========================
# 1️⃣ Parquet確認
# =========================
print("\n=========================")
print("📦 STEP 1: Parquet確認")
print("=========================")

if not os.path.exists(PARQUET_FILE):
    print("⚠ Parquet not found → 初回取得実行")
else:
    print("✅ Parquet exists")

# =========================
# 2️⃣ 株価データ更新
# =========================
print("\n=========================")
print("📊 STEP 2: データ更新")
print("=========================")

run_script("download_prices.py")

check_file_exists(PARQUET_FILE, "Parquet")
check_file_size(PARQUET_FILE, 1_000_000, "Parquet")

# =========================
# 3️⃣ 特徴量作成（強化版）
# =========================
print("\n=========================")
print("🧠 STEP 3: Feature Engineering")
print("=========================")

run_script("feature_engineering.py")

check_file_exists(TRAIN_FILE, "Train Dataset")
check_file_exists(PREDICT_FILE, "Predict Dataset")

check_file_size(TRAIN_FILE, 500_000, "Train Dataset")
check_file_size(PREDICT_FILE, 500_000, "Predict Dataset")

# =========================
# 4️⃣ 予測
# =========================
print("\n=========================")
print("🤖 STEP 4: Prediction")
print("=========================")

run_script("run_prediction.py")

# =========================
# 5️⃣ パフォーマンス計算
# =========================
print("\n=========================")
print("📈 STEP 5: Performance")
print("=========================")

run_script("calc_performance.py")

# =========================
# 完了
# =========================
print("\n=========================")
print("🎉 Pipeline finished successfully")
print("=========================")
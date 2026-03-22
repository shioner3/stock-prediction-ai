import subprocess
import os

PARQUET_FILE = "stock_data/prices.parquet"


def run_script(script_name):
    print(f"Running {script_name}...")
    script_path = os.path.join(os.getcwd(), script_name)
    subprocess.run(["python", script_path], check=True)


# =========================
# 1️⃣ Parquet初期化確認
# =========================
if not os.path.exists(PARQUET_FILE):
    print("Parquet not found. It will be created by download_prices.py")

# =========================
# 2️⃣ 株価データ取得（差分更新）
# =========================
run_script("download_prices.py")

# =========================
# 3️⃣ 特徴量作成
# =========================
run_script("feature_engineering.py")

# =========================
# 4️⃣ 予測
# =========================
run_script("run_prediction.py")

print("Pipeline finished.")

# =========================
# 5️⃣ Parquetサイズチェック
# =========================
size = os.path.getsize(PARQUET_FILE)
print("Parquet size:", size)

if size < 1000000:
    raise Exception("Parquet破損の可能性あり。保存停止。")

run_script("calc_performance.py")
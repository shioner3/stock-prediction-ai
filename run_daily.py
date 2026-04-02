import subprocess
import os
import time

PARQUET_FILE = "stock_data/prices.parquet"

def run_script(script_name):
    print(f"\n🚀 Running {script_name}...")
    script_path = os.path.join(os.getcwd(), script_name)

    try:
        subprocess.run(["python", script_path], check=True)
        print(f"✅ 完了: {script_name}")
    except Exception as e:
        print(f"❌ エラー: {script_name}")
        print(e)

# =========================
# 1️⃣ Parquet確認
# =========================
if not os.path.exists(PARQUET_FILE):
    print("⚠️ Parquetなし → 新規作成")

# =========================
# 2️⃣ 株価取得
# =========================
run_script("download_prices.py")

# =========================
# 3️⃣ 特徴量生成
# =========================
run_script("feature_engineering.py")

# =========================
# 4️⃣ 予測（最重要）
# =========================
run_script("run_prediction.py")

print("\n📊 Prediction finished.")

# =========================
# 5️⃣ Parquetサイズチェック
# =========================
if os.path.exists(PARQUET_FILE):
    size = os.path.getsize(PARQUET_FILE)
    print("Parquet size:", size)

    if size < 1000000:
        raise Exception("❌ Parquet破損の可能性")
else:
    raise Exception("❌ Parquet消失")

# =========================
# 6️⃣ 実績更新（前日の結果）
# =========================
run_script("calc_performance.py")

print("\n📈 Performance updated.")

# =========================
# 7️⃣ 記事生成（NEW）
# =========================
run_script("generate_article.py")

print("\n📝 Article generated.")

print("\n🎯 全処理完了")
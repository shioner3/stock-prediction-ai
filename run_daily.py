import subprocess
import os

# 現在のスクリプトのディレクトリを取得
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name):
    script_path = os.path.join(BASE_DIR, script_name)
    print(f"Running {script_name}...")
    subprocess.run(["python3", script_path], check=True)

# =========================
# 実行
# =========================
run_script("download_prices.py")
run_script("feature_engineering.py")
run_script("run_prediction.py")

print("Done.")

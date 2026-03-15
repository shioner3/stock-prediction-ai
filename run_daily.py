import subprocess
import os

def run_script(script_name):
    print(f"Running {script_name}...")
    script_path = os.path.join(os.getcwd(), script_name)
    subprocess.run(["python", script_path], check=True)

run_script("download_prices.py")
run_script("feature_engineering.py")
run_script("run_prediction.py")

print("Pipeline finished.")

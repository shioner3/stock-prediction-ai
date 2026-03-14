import os

print("Updating prices...")
os.system("python download_prices.py")

print("Creating features...")
os.system("python feature_engineering.py")

print("Running AI prediction...")
os.system("python run_prediction.py")

print("Done.")
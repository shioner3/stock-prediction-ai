import pandas as pd
import numpy as np
import pickle

# =========================
# 設定（AI主体・4日戦略）
# =========================
DATA_PATH = "ml_dataset_4d.parquet"
MODEL_PATH = "model.pkl"

HOLD_DAYS = 4
TOP_N = 3
TOP_RATE = 0.02

INITIAL_CAPITAL = 1.0
FEE = 0.001

# =========================
# データ
# =========================
df = pd.read_parquet(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

# =========================
# モデル読み込み
# =========================
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# =========================
# 特徴量
# =========================
FEATURES = [
    "Breakout",
    "Volume_Spike",
    "Vol_Expansion",
    "Gap",
    "final_score"
]

df = df.dropna(subset=FEATURES).copy()

# =========================
# 🔥 AI予測
# =========================
df["pred_score"] = model.predict(df[FEATURES])

# =========================
# 価格辞書
# =========================
price_open = {(r.Date, r.Ticker): r.Open for r in df.itertuples()}

# =========================
# 日付処理
# =========================
dates = sorted(df["Date"].unique())
date_groups = dict(tuple(df.groupby("Date")))

# =========================
# バックテスト
# =========================
capital = INITIAL_CAPITAL
positions = []
equity = []

for i in range(len(dates) - HOLD_DAYS - 1):

    today = dates[i]
    next_day = dates[i + 1]
    df_today = date_groups[today].copy()

    # =========================
    # EXIT
    # =========================
    daily_return = 0
    new_pos = []

    for p in positions:
        if i == p["exit_idx"]:
            price = price_open.get((today, p["Ticker"]))
            if price is not None:
                ret = (price / p["entry"] - 1) - FEE
                daily_return += ret * p["w"]
        else:
            new_pos.append(p)

    positions = new_pos

    # =========================
    # ENTRY（AI主体）
    # =========================
    df_today["pred_rank"] = df_today["pred_score"].rank(ascending=False, pct=True)

    candidates = df_today[df_today["pred_rank"] <= TOP_RATE]

    if len(candidates) == 0:
        capital *= (1 + daily_return)
        equity.append(capital)
        continue

    selected = candidates.sort_values("pred_score", ascending=False).head(TOP_N)

    # =========================
    # ポジション追加
    # =========================
    slots = TOP_N - len(positions)

    if slots > 0:
        entries = selected.head(slots)

        weights = np.exp(entries["pred_score"])
        weights /= weights.sum()

        for (_, r), w in zip(entries.iterrows(), weights):
            price = price_open.get((next_day, r["Ticker"]))
            if price is not None:
                positions.append({
                    "Ticker": r["Ticker"],
                    "entry": price * (1 + FEE),
                    "exit_idx": i + HOLD_DAYS,
                    "w": w
                })

    # =========================
    # 正規化
    # =========================
    if positions:
        s = sum(p["w"] for p in positions)
        for p in positions:
            p["w"] /= s

    # =========================
    # 資産更新
    # =========================
    capital *= (1 + daily_return)
    equity.append(capital)

# =========================
# 評価
# =========================
equity = pd.Series(equity)

ret = equity.pct_change().fillna(0)

CAGR = equity.iloc[-1] ** (252 / len(equity)) - 1
Sharpe = ret.mean() / (ret.std() + 1e-9) * np.sqrt(252)
MaxDD = (equity / equity.cummax() - 1).min()

print("\n=== RESULT（AI主体・4日戦略） ===")
print(f"CAGR  : {CAGR:.4f}")
print(f"Sharpe: {Sharpe:.4f}")
print(f"MaxDD : {MaxDD:.4f}")
import pandas as pd
import numpy as np
import itertools

from signal_engine import generate_signals

# =========================
# データ
# =========================
df = pd.read_parquet("stock_data/features.parquet")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Ticker"])

# =========================
# パラメータ空間
# =========================
param_grid = {
    "ret_th": [0.0, 0.01, 0.02, 0.03],
    "vol_th": [1.0, 1.1, 1.2, 1.5],
    "ma_th": [0.0, 0.05, 0.1],
    "tp": [0.10, 0.15, 0.20, 0.25],
    "sl": [-0.02, -0.03, -0.05, -0.08],
    "hold": [5, 10, 15, 20],
}

results = []

# =========================
# グリッド
# =========================
for ret_th, vol_th, ma_th, tp, sl, hold in itertools.product(
    param_grid["ret_th"],
    param_grid["vol_th"],
    param_grid["ma_th"],
    param_grid["tp"],
    param_grid["sl"],
    param_grid["hold"]
):

    d = df.copy()

    # =========================
    # シグナル生成
    # =========================
    d = d[
        (d["return_3d"] > ret_th) &
        (d["volume_ratio"] > vol_th) &
        (d["ma5_diff"] < ma_th) &
        (d["market_trend_5"] > -0.005)
    ]

    if len(d) < 100:
        continue

    d["signal_score"] = (
        d["return_3d"] * 10 +
        d["volume_ratio"] +
        d["return_rank"]
    )

    d = d.sort_values(["Date", "signal_score"], ascending=[True, False])
    d = d.groupby("Date").head(1)

    # =========================
    # exit計算
    # =========================
    g = df.groupby("Ticker")

    returns = []

    for _, row in d.iterrows():

        t = row["Ticker"]
        date = row["Date"]

        df_t = g.get_group(t)

        idx = df_t.index[df_t["Date"] == date]
        if len(idx) == 0:
            continue

        i = idx[0]

        entry_price = df_t.iloc[i+1]["Open"]

        exit_price = entry_price
        entry_ret = None

        for j in range(1, hold):

            if i + j >= len(df_t):
                break

            price = df_t.iloc[i+j]["Close"]

            r = price / entry_price - 1

            if r > tp:
                exit_price = price
                break

            if r < sl:
                exit_price = price
                break

        ret = exit_price / entry_price - 1
        returns.append(ret)

    if len(returns) < 50:
        continue

    returns = np.array(returns)

    sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)

    results.append({
        "ret_th": ret_th,
        "vol_th": vol_th,
        "ma_th": ma_th,
        "tp": tp,
        "sl": sl,
        "hold": hold,
        "sharpe": sharpe,
        "n": len(returns)
    })

# =========================
# 結果
# =========================
res_df = pd.DataFrame(results)
res_df = res_df.sort_values("sharpe", ascending=False)

print("\n=== TOP RESULT ===")
print(res_df.head(10))
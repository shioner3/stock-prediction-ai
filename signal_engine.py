import pandas as pd
import numpy as np

TOP_N = 1

def generate_signals(df):

    df = df.copy()

    # ===== 超緩い条件 =====
    df = df[
        (df["return_3d"] > 0.01) &
        (df["volume_ratio"] > 1.1)
    ]

    # ===== スコア =====
    df["signal_score"] = (
        df["return_3d"] * 10 +
        np.log1p(df["volume_ratio"]) * 2 +
        df["return_rank"] * 2 -
        df["volatility_5"] * 3 -
        df["ma5_diff"] * 2
    )

    # ===== ランク =====
    df["rank"] = df.groupby("Date")["signal_score"]\
        .rank(ascending=False, method="first")

    df["signal"] = df["rank"] <= TOP_N

    return df
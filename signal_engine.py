import pandas as pd
import numpy as np

TOP_N = 2

def generate_signals(df):

    df = df.copy()

    # ===== 初動検出 =====
    df = df[
        (df["return_3d"] > 0) &
        (df["return_3d"] < 0.02) &
        (df["volume_ratio"] > 1.3) &
        (df["ma5_diff"] < 0.02)
    ]

    # ===== スコア =====
    df["signal_score"] = (
        df["volume_ratio"] * 2 +
        df["return_rank"] * 2 -
        df["ma5_diff"] * 3 -
        df["volatility_5"] * 3
    )

    # ===== ランク =====
    df["rank"] = df.groupby("Date")["signal_score"]\
        .rank(ascending=False, method="first")

    df["signal"] = df["rank"] <= TOP_N

    return df
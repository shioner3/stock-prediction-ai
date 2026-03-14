import streamlit as st
import pandas as pd
import os

st.title("毎日更新♪株予測AI(´▽｀)")

DATA_PATH = "today_picks.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    expected_cols = ["コード", "銘柄名", "PredRank"]
    if all(col in df.columns for col in expected_cols):
        st.subheader("5日後に上がりそうな5銘柄")
        df_display = df[expected_cols].copy()
        df_display = df_display.rename(columns={"PredRank": "順位"})
        st.dataframe(df_display)
    else:
        st.warning(f"CSV に必要な列がありません: {df.columns.tolist()}")
else:
    st.warning("CSV がまだ生成されていません。Run Daily を先に実行してください。")

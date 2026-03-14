import streamlit as st
import pandas as pd
import os

st.title("株予測AI")

# today_picks.csv を読み込む
DATA_PATH = "today_picks.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    
    st.subheader("5日後に上がりそうな5銘柄")
    
    # Streamlit で表示列を整理
    # 「コード」「銘柄名」「PredRank」
    df_display = df[["コード", "銘柄名", "PredRank"]].copy()
    df_display = df_display.rename(columns={
        "コード": "コード",
        "銘柄名": "銘柄名",
        "PredRank": "順位"
    })
    
    st.dataframe(df_display)
else:
    st.warning("today_picks.csv がまだ生成されていません")

import streamlit as st
import pandas as pd
import os

st.title("毎日更新♪株予測AI(*^^*)")

if os.path.exists("today_picks.csv"):
    df = pd.read_csv("today_picks.csv")
    st.subheader("5日後に上がりそうな5銘柄")
    st.dataframe(df)
else:
    st.warning("CSV がまだ生成されていません。Run Daily を先に実行してください。")

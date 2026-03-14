import streamlit as st
import pandas as pd

st.title("5日後に上昇していそうな株")

df = pd.read_csv("today_picks.csv")

st.subheader("今日のおすすめ銘柄")

st.dataframe(df)

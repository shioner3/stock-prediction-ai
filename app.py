import streamlit as st
import pandas as pd

st.title("株予測AI")

df = pd.read_csv("today_picks.csv")

st.subheader("5日後に上がりそうな5銘柄")

st.dataframe(df)

import streamlit as st
import yfinance as yf

st.title("Test Yahoo Finance")

ticker = st.text_input("Enter ticker", "AAPL")

if st.button("Fetch"):
    data = yf.download(ticker, start="2022-01-01", end="2022-12-31")
    st.write(data.head())

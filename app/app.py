import streamlit as st

st.set_page_config(
    page_title="LifeSaverBN3000",
    page_icon="🩺",
    layout="wide"
)
st.title("LifeSaverBN3000 — Obesity Risk Estimation")
st.markdown("Welcome to LifeSaverBN3000! This application estimates the risk of obesity based on various health and lifestyle factors. Please navigate through the sections to input your data and receive personalized insights.")
st.sidebar.header("Patient Information")
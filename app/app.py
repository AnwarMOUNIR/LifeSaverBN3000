import streamlit as st

st.set_page_config(
    page_title="LifeSaverBN3000",
    page_icon="🩺",
    layout="wide"
)
st.title("LifeSaverBN3000 — Obesity Risk Estimation")
st.markdown("Welcome to LifeSaverBN3000! This application estimates the risk of obesity based on various health and lifestyle factors. Please navigate through the sections to input your data and receive personalized insights.")
st.sidebar.header("Patient Information")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25, step=1)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
height = st.sidebar.number_input("Height (m)", min_value=1.0, max_value=2.5,value=1.70, step=0.01, format="%.2f")
weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=250.0,value=70.0, step=0.5)
bmi = weight / (height ** 2)
st.sidebar.markdown(f"**Calculated BMI:** {bmi:.2f}")
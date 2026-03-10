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
# ── Family History ──
st.sidebar.subheader("Family & History")

family_history = st.sidebar.selectbox(
    "Family history of overweight?",
    options=["yes", "no"]
)

# ── Eating Habits ──
st.sidebar.subheader("Eating Habits")

favc = st.sidebar.selectbox(
    "Frequent high-calorie food consumption?",
    options=["yes", "no"]
)

fcvc = st.sidebar.slider(
    "Vegetable consumption frequency",
    min_value=1.0, max_value=3.0, value=2.0, step=0.1
)

ncp = st.sidebar.slider(
    "Number of main meals per day",
    min_value=1.0, max_value=4.0, value=3.0, step=0.1
)

caec = st.sidebar.selectbox(
    "Food between meals?",
    options=["no", "Sometimes", "Frequently", "Always"]
)

ch2o = st.sidebar.slider(
    "Daily water intake (liters)",
    min_value=1.0, max_value=3.0, value=2.0, step=0.1
)

calc = st.sidebar.selectbox(
    "Alcohol consumption?",
    options=["no", "Sometimes", "Frequently", "Always"]
)

# ── Physical Condition ──
st.sidebar.subheader("Physical Condition")

smoke = st.sidebar.radio("Do you smoke?", options=["yes", "no"], horizontal=True)

scc = st.sidebar.radio("Do you monitor daily calories?", options=["yes", "no"], horizontal=True)

faf = st.sidebar.slider(
    "Physical activity frequency (days/week)",
    min_value=0.0, max_value=3.0, value=1.0, step=0.1
)

tue = st.sidebar.slider(
    "Time using technology devices (hours/day)",
    min_value=0.0, max_value=2.0, value=1.0, step=0.1
)

mtrans = st.sidebar.selectbox(
    "Main mode of transportation",
    options=["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]
)
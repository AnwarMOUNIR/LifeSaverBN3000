import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="LifeSaverBN3000",
    page_icon="🩺",
    layout="wide"
)

# ── Paths ──
APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(APP_DIR)
MODEL_PATH = os.path.join(REPO_ROOT, "models", "best_model.pkl")
PROCESSED_DATA_PATH = os.path.join(REPO_ROOT, "data", "processed", "processed_data.csv")
RAW_DATA_PATH = os.path.join(REPO_ROOT, "data", "ObesityDataSet_raw_and_data_sinthetic.csv")

# ── Load Model & Feature Columns ──
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def get_feature_columns():
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)
        return df.columns[:-1].tolist()
    return None

model = load_model()
feature_cols = get_feature_columns()

# ── Header ──
st.title("LifeSaverBN3000 — Obesity Risk Estimation")
st.markdown(
    "Welcome to LifeSaverBN3000! This application estimates the risk of obesity based on "
    "various health and lifestyle factors. Please fill in the sidebar and click **Run Prediction**."
)

# ── Sidebar: Patient Information ──
st.sidebar.header("Patient Information")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25, step=1)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
height = st.sidebar.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01, format="%.2f")
weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.5)
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

# ── Raw Input DataFrame ──
input_data = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "Height": [height],
    "Weight": [weight],
    "family_history_with_overweight": [family_history],
    "FAVC": [favc],
    "FCVC": [fcvc],
    "NCP": [ncp],
    "CAEC": [caec],
    "SMOKE": [smoke],
    "CH2O": [ch2o],
    "SCC": [scc],
    "FAF": [faf],
    "TUE": [tue],
    "CALC": [calc],
    "MTRANS": [mtrans],
})

st.subheader("Input Data Summary")
st.dataframe(input_data)


# ── Encoding ──
def encode_input(input_df, feature_cols):
    """
    Encode a single-row input DataFrame to match the training encoding.

    The training pipeline used pd.get_dummies(df, drop_first=True) on the full
    raw dataset. To replicate the same encoding for a new row we build the
    feature vector manually using the same drop-first convention, then align
    it with the exact column list the model was trained on.
    """
    row = {
        # Numeric columns (passed through unchanged)
        "Age": input_df["Age"].values[0],
        "Height": input_df["Height"].values[0],
        "Weight": input_df["Weight"].values[0],
        "FCVC": input_df["FCVC"].values[0],
        "NCP": input_df["NCP"].values[0],
        "CH2O": input_df["CH2O"].values[0],
        "FAF": input_df["FAF"].values[0],
        "TUE": input_df["TUE"].values[0],
        # Binary dummies (drop_first removes the first alphabetical category)
        "Gender_Male": 1 if input_df["Gender"].values[0] == "Male" else 0,
        "family_history_with_overweight_yes": 1 if input_df["family_history_with_overweight"].values[0] == "yes" else 0,
        "FAVC_yes": 1 if input_df["FAVC"].values[0] == "yes" else 0,
        "SMOKE_yes": 1 if input_df["SMOKE"].values[0] == "yes" else 0,
        "SCC_yes": 1 if input_df["SCC"].values[0] == "yes" else 0,
        # CAEC: Always is dropped (first alphabetically); remaining are Frequently, Sometimes, no
        "CAEC_Frequently": 1 if input_df["CAEC"].values[0] == "Frequently" else 0,
        "CAEC_Sometimes": 1 if input_df["CAEC"].values[0] == "Sometimes" else 0,
        "CAEC_no": 1 if input_df["CAEC"].values[0] == "no" else 0,
        # CALC: same structure as CAEC
        "CALC_Frequently": 1 if input_df["CALC"].values[0] == "Frequently" else 0,
        "CALC_Sometimes": 1 if input_df["CALC"].values[0] == "Sometimes" else 0,
        "CALC_no": 1 if input_df["CALC"].values[0] == "no" else 0,
        # MTRANS: Automobile is dropped; remaining are Bike, Motorbike, Public_Transportation, Walking
        "MTRANS_Bike": 1 if input_df["MTRANS"].values[0] == "Bike" else 0,
        "MTRANS_Motorbike": 1 if input_df["MTRANS"].values[0] == "Motorbike" else 0,
        "MTRANS_Public_Transportation": 1 if input_df["MTRANS"].values[0] == "Public_Transportation" else 0,
        "MTRANS_Walking": 1 if input_df["MTRANS"].values[0] == "Walking" else 0,
    }

    encoded = pd.DataFrame([row])
    # reindex aligns columns and fills any training-only columns (e.g. NObeyesdad
    # leakage columns) with 0, which is the correct value for an unseen patient.
    encoded = encoded.reindex(columns=feature_cols, fill_value=0)
    return encoded


# ── Prediction Section ──
st.subheader("Obesity Risk Prediction")

if model is None:
    st.warning(
        "No trained model found at `models/best_model.pkl`. "
        "Please run `src/train_model.py` first."
    )
elif feature_cols is None:
    st.warning(
        "No processed data found at `data/processed/processed_data.csv`. "
        "Please run `src/data_processing.py` first."
    )
else:
    if st.button("Run Prediction", type="primary"):
        encoded_input = encode_input(input_data, feature_cols)
        prediction = model.predict(encoded_input)[0]
        probability = model.predict_proba(encoded_input)[0][1]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("BMI", f"{bmi:.2f}")
        with col2:
            st.metric("Model Risk Score", f"{probability:.1%}")

        if prediction == 1:
            st.error(
                f"**Result: Elevated Risk (Overweight Level II)**  \n"
                f"The model indicates a high probability ({probability:.1%}) of Overweight Level II."
            )
        else:
            st.success(
                f"**Result: Lower Risk**  \n"
                f"The model indicates a lower probability ({probability:.1%}) of Overweight Level II."
            )

        with st.expander("Encoded feature vector sent to model"):
            st.dataframe(encoded_input)

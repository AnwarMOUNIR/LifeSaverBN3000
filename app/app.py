import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

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

# ── Load Model, Feature Columns & SHAP Explainer ──
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

@st.cache_resource
def get_shap_explainer(_model):
    """Build a TreeExplainer (works for RF, XGBoost, LightGBM, CatBoost)."""
    return shap.TreeExplainer(_model)

@st.cache_data
def get_global_shap_values(cache_bust=1):
    """
    Compute SHAP values on a sample of the training data for the global
    summary plot. Results are cached so this only runs once.
    """
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.iloc[:, :-1]
    # Sample for speed; adjust size as needed
    X_sample = X.sample(min(300, len(X)), random_state=42)
    explainer = get_shap_explainer(load_model())
    raw = explainer.shap_values(X_sample)
    
    # SHAP will naturally render a stacked bar plot for lists (multiclass)
    return raw, X_sample

model = load_model()
feature_cols = get_feature_columns()

# ── Header ──
st.title("LifeSaverBN3000 — Obesity Risk Estimation")
st.markdown(
    "Welcome to LifeSaverBN3000! This application estimates the risk of obesity based on "
    "various health and lifestyle factors. Fill in the sidebar and click **Run Prediction**."
)

# ── Sidebar: Patient Information ──
st.sidebar.header("Patient Information")
age    = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25, step=1)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
height = st.sidebar.number_input("Height (m)", min_value=1.0, max_value=2.5,
                                  value=1.70, step=0.01, format="%.2f")
weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=250.0,
                                  value=70.0, step=0.5)
bmi = weight / (height ** 2)
st.sidebar.markdown(f"**Calculated BMI:** {bmi:.2f}")

st.sidebar.subheader("Family & History")
family_history = st.sidebar.selectbox("Family history of overweight?", options=["yes", "no"])

st.sidebar.subheader("Eating Habits")
favc = st.sidebar.selectbox("Frequent high-calorie food consumption?", options=["yes", "no"])
fcvc = st.sidebar.slider("Vegetable consumption frequency",
                          min_value=1.0, max_value=3.0, value=2.0, step=0.1)
ncp  = st.sidebar.slider("Number of main meals per day",
                          min_value=1.0, max_value=4.0, value=3.0, step=0.1)
caec = st.sidebar.selectbox("Food between meals?",
                             options=["no", "Sometimes", "Frequently", "Always"])
ch2o = st.sidebar.slider("Daily water intake (liters)",
                          min_value=1.0, max_value=3.0, value=2.0, step=0.1)
calc = st.sidebar.selectbox("Alcohol consumption?",
                             options=["no", "Sometimes", "Frequently", "Always"])

st.sidebar.subheader("Physical Condition")
smoke  = st.sidebar.radio("Do you smoke?", options=["yes", "no"], horizontal=True)
scc    = st.sidebar.radio("Do you monitor daily calories?", options=["yes", "no"], horizontal=True)
faf    = st.sidebar.slider("Physical activity frequency (days/week)",
                            min_value=0.0, max_value=3.0, value=1.0, step=0.1)
tue    = st.sidebar.slider("Time using technology devices (hours/day)",
                            min_value=0.0, max_value=2.0, value=1.0, step=0.1)
mtrans = st.sidebar.selectbox("Main mode of transportation",
                               options=["Automobile", "Bike", "Motorbike",
                                        "Public_Transportation", "Walking"])

# ── Raw Input DataFrame ──
input_data = pd.DataFrame({
    "Gender": [gender], "Age": [age], "Height": [height], "Weight": [weight],
    "family_history_with_overweight": [family_history], "FAVC": [favc],
    "FCVC": [fcvc], "NCP": [ncp], "CAEC": [caec], "SMOKE": [smoke],
    "CH2O": [ch2o], "SCC": [scc], "FAF": [faf], "TUE": [tue],
    "CALC": [calc], "MTRANS": [mtrans], "BMI": [bmi]
})


# ── Encoding ──
def encode_input(input_df, feature_cols):
    """
    Replicate pd.get_dummies(df, drop_first=True) encoding used during training.
    Uses reindex to align columns and fill any unseen columns (e.g. leaked
    NObeyesdad dummies) with 0.
    """
    row = {
        "Age": input_df["Age"].values[0],
        "Height": input_df["Height"].values[0],
        "Weight": input_df["Weight"].values[0],
        "BMI": input_df["BMI"].values[0],
        "FCVC": input_df["FCVC"].values[0],
        "NCP": input_df["NCP"].values[0],
        "CH2O": input_df["CH2O"].values[0],
        "FAF": input_df["FAF"].values[0],
        "TUE": input_df["TUE"].values[0],
        "Gender_Male": 1 if input_df["Gender"].values[0] == "Male" else 0,
        "family_history_with_overweight_yes":
            1 if input_df["family_history_with_overweight"].values[0] == "yes" else 0,
        "FAVC_yes":  1 if input_df["FAVC"].values[0]  == "yes" else 0,
        "SMOKE_yes": 1 if input_df["SMOKE"].values[0] == "yes" else 0,
        "SCC_yes":   1 if input_df["SCC"].values[0]   == "yes" else 0,
        # CAEC — Always is the dropped baseline
        "CAEC_Frequently": 1 if input_df["CAEC"].values[0] == "Frequently" else 0,
        "CAEC_Sometimes":  1 if input_df["CAEC"].values[0] == "Sometimes"  else 0,
        "CAEC_no":         1 if input_df["CAEC"].values[0] == "no"         else 0,
        # CALC — same structure
        "CALC_Frequently": 1 if input_df["CALC"].values[0] == "Frequently" else 0,
        "CALC_Sometimes":  1 if input_df["CALC"].values[0] == "Sometimes"  else 0,
        "CALC_no":         1 if input_df["CALC"].values[0] == "no"         else 0,
        # MTRANS — Automobile is the dropped baseline
        "MTRANS_Bike":                  1 if input_df["MTRANS"].values[0] == "Bike"                  else 0,
        "MTRANS_Motorbike":             1 if input_df["MTRANS"].values[0] == "Motorbike"             else 0,
        "MTRANS_Public_Transportation": 1 if input_df["MTRANS"].values[0] == "Public_Transportation" else 0,
        "MTRANS_Walking":               1 if input_df["MTRANS"].values[0] == "Walking"               else 0,
    }
    encoded = pd.DataFrame([row])
    encoded = encoded.reindex(columns=feature_cols, fill_value=0)
    return encoded


# ════════════════════════════════════════════════════════
# Main Page Content
# ════════════════════════════════════════════════════════

st.subheader("Input Data Summary")
st.dataframe(input_data)

st.subheader("Obesity Risk Prediction")

if model is None:
    st.warning("No trained model found at `models/best_model.pkl`. "
               "Please run `src/train_model.py` first.")
elif feature_cols is None:
    st.warning("No processed data found at `data/processed/processed_data.csv`. "
               "Please run `src/data_processing.py` first.")
else:
    if st.button("Run Prediction", type="primary"):
        encoded_input = encode_input(input_data, feature_cols)
        prediction  = model.predict(encoded_input)[0]
        # Get the probability of the predicted class (it's multiclass!)
        probability = model.predict_proba(encoded_input)[0][prediction]
        
        # Load the Label Encoder to get the semantic string mapping
        le_path = os.path.join(REPO_ROOT, "models", "label_encoder.pkl")
        if os.path.exists(le_path):
            le = joblib.load(le_path)
            prediction_label = le.inverse_transform([prediction])[0]
        else:
            prediction_label = str(prediction)

        # Store in session state so the Explainability tab can access them
        st.session_state["encoded_input"] = encoded_input
        st.session_state["prediction_label"] = prediction_label
        st.session_state["probability"]   = probability

    # Show result if we have one (persists across reruns)
    if "prediction_label" in st.session_state:
        prediction_label  = st.session_state["prediction_label"]
        probability = st.session_state["probability"]
        encoded_input = st.session_state["encoded_input"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("BMI", f"{bmi:.2f}")
        with col2:
            st.metric("Model Confidence", f"{probability:.1%}")

        if "Normal_Weight" in prediction_label or "Insufficient" in prediction_label:
            st.success(
                f"**Result: {prediction_label.replace('_', ' ')}**  \n"
                f"The model assigns a {probability:.1%} confidence."
            )
        else:
            st.error(
                f"**Result: {prediction_label.replace('_', ' ')}**  \n"
                f"The model assigns a {probability:.1%} confidence."
            )

        # ── Individual SHAP Waterfall Plot ──
        st.subheader("Individual Prediction Explanation")
        st.caption(
            "The waterfall chart shows how each feature pushed the prediction "
            "above (red) or below (blue) the model's baseline."
        )
        with st.spinner("Computing individual SHAP values…"):
            explainer = get_shap_explainer(model)
            raw_ind   = explainer.shap_values(encoded_input)

            if isinstance(raw_ind, list):
                # Use the specific predicted class index
                sv = raw_ind[prediction][0]
                bv = explainer.expected_value[prediction]
            else:
                sv = raw_ind[0]
                bv = (explainer.expected_value[0]
                      if hasattr(explainer.expected_value, "__len__")
                      else explainer.expected_value)

            ind_exp = shap.Explanation(
                values=sv,
                base_values=bv,
                data=encoded_input.values[0],
                feature_names=list(encoded_input.columns),
            )

            fig_ind, ax_ind = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(ind_exp, max_display=10, show=False)
            fig_ind = plt.gcf()
            st.pyplot(fig_ind, bbox_inches="tight")
            plt.close("all")

        with st.expander("Encoded feature vector sent to model"):
            st.dataframe(encoded_input)

st.markdown("---")
# ════════════════════════════════════════════════════════
# Global Explainability
# ════════════════════════════════════════════════════════
st.subheader("Global Feature Importance — SHAP Summary Plot")
st.markdown(
    "Each dot represents one training sample. Features are ranked by their "
    "average impact on the model output. **Red** = high feature value, "
    "**Blue** = low feature value."
)

if model is None or feature_cols is None:
    st.warning("Model or processed data not found. Please train the model first.")
else:
    with st.spinner("Computing global SHAP values (this may take a moment)…"):
        global_shap_vals, X_sample = get_global_shap_values(cache_bust=2)

    fig_global, ax_global = plt.subplots(figsize=(10, 6))
    shap.summary_plot(global_shap_vals, X_sample, plot_size=(10, 6), max_display=12, show=False)
    fig_global = plt.gcf()
    st.pyplot(fig_global, bbox_inches="tight")
    plt.close("all")

    st.caption(
        f"Summary computed on a random sample of {len(X_sample)} training rows. "
        "Re-run `src/train_model.py` to refresh the model."
    )

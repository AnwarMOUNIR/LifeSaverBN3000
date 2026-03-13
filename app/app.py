import streamlit as st
import pandas as pd
import joblib
import os
import sys
from sklearn.pipeline import Pipeline
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

# Add REPO_ROOT to sys.path to allow imports from src/
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.train_model import apply_sanity_guards

# ── Load Model, Feature Columns & SHAP Explainer ──
def load_model():
    if os.path.exists(MODEL_PATH):
        obj = joblib.load(MODEL_PATH)
        # Diagnostic print for Streamlit Cloud logs
        print(f"DEBUG: Loaded model type: {type(obj)}")
        if hasattr(obj, 'steps'):
            print(f"DEBUG: Pipeline steps: {[s[0] for s in obj.steps]}")
            if hasattr(obj.steps[0][1], 'feature_names_in_'):
                print(f"DEBUG: Expected input features: {obj.steps[0][1].feature_names_in_}")
        return obj
    return None

def get_feature_columns():
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)
        return df.columns[:-1].tolist()
    return None

def get_shap_explainer(_model):
    """Build a TreeExplainer (works for RF, XGBoost, LightGBM, CatBoost)."""
    # If it's a pipeline, extract the actual estimator (the last step)
    if hasattr(_model, 'steps'):
        actual_model = _model.steps[-1][1]
    else:
        actual_model = _model
    return shap.TreeExplainer(actual_model)

def get_global_shap_values(_model):
    """
    Compute SHAP values on a sample of the training data for the global
    summary plot. Results are cached so this only runs once.
    """
    # Use raw data for global SHAP to ensure consistency with pipeline
    raw_data_path = os.path.join(REPO_ROOT, "data", "raw", "ObesityDataSet_synthetic.csv")
    if not os.path.exists(raw_data_path):
        return None, None
        
    try:
        df = pd.read_csv(raw_data_path)
        X = df.iloc[:, :-1]
        # Sample for speed
        X_sample_raw = X.sample(min(300, len(X)), random_state=42)
        
        # We must encode the raw sample using the model's preprocessor
        if hasattr(_model, 'steps'):
            preprocessing_pipe = Pipeline(_model.steps[:-1])
            X_sample_encoded = preprocessing_pipe.transform(X_sample_raw)
        else:
            X_sample_encoded = X_sample_raw

        explainer = get_shap_explainer(_model)
        raw = explainer.shap_values(X_sample_encoded)
        
        return raw, X_sample_encoded
    except Exception as e:
        print(f"ERROR: Global SHAP computation failed: {e}")
        return None, None

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
    "CALC": [calc], "MTRANS": [mtrans]
})


# ── Encoding ──
def encode_input(input_df, feature_cols):
    """
    (Deprecated) The new best_model.pkl is a Pipeline that handles 
    engineering and encoding internally. 
    """
    return input_df


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
        # Clear previous session state for fresh prediction
        for key in ["prediction_label", "raw_model_label", "probability", "is_guard_active", "encoded_input", "input_data_bmi"]:
            if key in st.session_state:
                del st.session_state[key]
                
        # The new pipeline handles Encoding internally
        prediction_val = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][prediction_val]
        
        # Load the Label Encoder to get the semantic string mapping
        le_path = os.path.join(REPO_ROOT, "models", "label_encoder.pkl")
        le = joblib.load(le_path) if os.path.exists(le_path) else None
        
        prediction_label = le.inverse_transform([prediction_val])[0] if le else str(prediction_val)

        # Apply Safety Guards
        final_label = apply_sanity_guards(input_data, prediction_label)
        
        is_guard_active = (final_label != prediction_label)

        # Store in session state
        st.session_state["prediction_label"] = final_label
        st.session_state["raw_model_label"] = prediction_label
        st.session_state["probability"] = probability
        st.session_state["is_guard_active"] = is_guard_active
        
        # For SHAP, we need the encoded features
        # We can extract them by running just the preprocessing part of the pipeline
        preprocessing_pipe = Pipeline(model.steps[:-1])
        st.session_state["encoded_input"] = preprocessing_pipe.transform(input_data)
        st.session_state["prediction_val"] = prediction_val
        st.session_state["input_data_bmi"] = weight / (height ** 2)

    # Show result if we have one (persists across reruns)
    if "prediction_label" in st.session_state:
        prediction_label  = st.session_state["prediction_label"]
        probability = st.session_state["probability"]
        encoded_input = st.session_state["encoded_input"]
        if is_guard_active:
            st.warning(f"🛡️ **Safety Guard Active**: The ML model predicted '{st.session_state['raw_model_label'].replace('_', ' ')}', but physiological rules overrode this to ensure medical accuracy.")
        
        # Diagnostic columns
        c1, c2 = st.columns(2)
        c1.metric("Model Confidence", f"{probability:.1%}")
        c2.metric("Calculated BMI", f"{bmi:.2f}")

        if "Normal_Weight" in prediction_label or "Insufficient" in prediction_label:
            st.success(
                f"**Final Result: {prediction_label.replace('_', ' ')}**"
            )
        else:
            st.error(
                f"**Final Result: {prediction_label.replace('_', ' ')}**"
            )
        
        st.info(f"💡 **Diagnostic Trace**:")
        st.code(f"Original Model Prediction: {st.session_state.get('raw_model_label', '')}\n"
                f"Applied Overridden Label: {prediction_label}\n"
                f"Guard Triggered: {is_guard_active}\n"
                f"Calculated BMI: {st.session_state.get('input_data_bmi', 0):.2f}")

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
                prediction_val = st.session_state["prediction_val"]
                sv = raw_ind[prediction_val][0]
                bv = explainer.expected_value[prediction_val]
            else:
                # Scikit-learn/XGBoost Multiclass returns (samples, features, classes) or similar
                prediction_val = st.session_state["prediction_val"]
                if len(getattr(raw_ind, "shape", [])) == 3:
                    # Case for 3D array (e.g., XGBoost)
                    sv = raw_ind[0, :, prediction_val]
                    bv = explainer.expected_value[prediction_val]
                elif len(getattr(raw_ind, "shape", [])) == 2 and raw_ind.shape[0] == 1:
                    # Case for 2D array (samples, features) for a single class or binary
                    sv = raw_ind[0]
                    bv = (explainer.expected_value[prediction_val] 
                          if hasattr(explainer.expected_value, "__len__")
                          else explainer.expected_value)
                else:
                    # Fallback
                    sv = raw_ind[0]
                    bv = explainer.expected_value

            ind_exp = shap.Explanation(
                values=sv,
                base_values=bv,
                data=encoded_input.values[0] if hasattr(encoded_input, "values") else encoded_input[0],
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
st.subheader("Global Feature Importance")
if model is None or feature_cols is None:
    st.warning("Model or processed data not found. Please train the model first.")
else:
    with st.spinner("Computing global SHAP values (this may take a moment)…"):
        global_shap_vals, X_sample = get_global_shap_values(model)

    if global_shap_vals is None or X_sample is None:
        st.warning("⚠️ **Global Explanation Unavailable**: The raw dataset was not found or is incompatible. Make sure `data/raw/ObesityDataSet_synthetic.csv` is present.")
    else:
        tab1, tab2 = st.tabs(["Feature Ranking (Simple)", "Detailed Impact (Summary Plot)"])
    
    with tab1:
        st.markdown(
            "This definitive ranking displays the average magnitude of influence each parameter has. "
            "Longer bars mean the parameter heavily dictates the obesity risk prediction across all patients."
        )
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        shap.summary_plot(global_shap_vals, X_sample, plot_type="bar", plot_size=(10, 6), max_display=12, show=False)
        fig_bar = plt.gcf()
        st.pyplot(fig_bar, bbox_inches="tight")
        plt.close("all")

    with tab2:
        st.markdown(
            "Each dot represents one training sample. Features are ranked vertically by importance. "
            "**Red** = high feature value, **Blue** = low feature value."
        )
        fig_global, ax_global = plt.subplots(figsize=(10, 6))
        shap.summary_plot(global_shap_vals, X_sample, plot_size=(10, 6), max_display=12, show=False)
        fig_global = plt.gcf()
        st.pyplot(fig_global, bbox_inches="tight")
        plt.close("all")

    st.caption(
        f"Summary computed on a random sample of {len(X_sample)} training rows. "
        "Re-run `src/train_model.py` to refresh the model."
    )

st.markdown("---")
# ── Credits ──
with st.expander("ℹ️ Meet the Team"):
    st.markdown("""
    <div style="text-align: center;">
    
    **Anwar Mounir**  
    *Project Lead & DevOps*
    
    **Wissal Ait Ali**  
    *Foundation Architect*
    
    **Zaid Kasimi**  
    *UI/UX Architect*
    
    **Ihsane Abdou**  
    *ML Developer*
    
    **Ikram Boulouqat**  
    *XAI & Performance*
    
    **Meryem Soussi**  
    *QA & Safety Specialist*
    
    </div>
    """, unsafe_allow_html=True)

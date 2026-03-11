"""Generate SHAP values and a global summary plot for the obesity classifier.

This script:
- Loads the best trained classification model (multiclass, 7 obesity classes)
  from ``models/best_model.pkl``.
- Loads the obesity dataset from ``data/ObesityDataSet_raw_and_data_synthetic.csv``.
- Builds an input feature matrix ``X`` by dropping the target column
  ``NObeyesdad`` and takes a random sample (up to 500 rows) for faster SHAP
  computation.
- Creates a SHAP explainer adapted to tree-based models (``TreeExplainer``).
- Computes SHAP values on the sampled test data.
- Generates and saves a SHAP summary beeswarm plot to
  ``outputs/plots/shap_summary_beeswarm.png``.

Run this file as a standalone script to regenerate the SHAP explanation plot:

    python -m src.evaluate_model
"""

from pathlib import Path
import os
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import numpy as np


MODEL_PATH = Path("models") / "best_model.pkl"
DATA_PATH = Path("data") / "ObesityDataSet_raw_and_data_synthetic.csv"
OUTPUT_DIR = Path("outputs") / "plots"
OUTPUT_PLOT_PATH = OUTPUT_DIR / "shap_summary_beeswarm.png"
TARGET_COLUMN = "NObeyesdad"
MAX_SAMPLE_SIZE = 500


def load_model(model_path: Path):
    """Load the trained model from disk using joblib."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Make sure the best model is trained and saved before running SHAP."
        )
    return joblib.load(model_path)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the obesity dataset used for model training/evaluation."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_path}. "
            "Please ensure the dataset is available before running SHAP."
        )
    return pd.read_csv(data_path)


def prepare_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Prepare the feature matrix X by dropping the target column."""
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found in DataFrame. "
            "Update TARGET_COLUMN to match the training setup."
        )
    X = df.drop(columns=[target_column])
    return X


def sample_data(X: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Sample up to max_rows rows from X for faster SHAP computation."""
    if len(X) == 0:
        raise ValueError("Input feature matrix X is empty; cannot compute SHAP values.")
    n_sample = min(max_rows, len(X))
    return X.sample(n=n_sample, random_state=42)


def create_explainer(model, X_sample: pd.DataFrame) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer for a tree-based multiclass classifier."""
    # For XGBoost / LightGBM / CatBoost / RandomForest and other tree models,
    # TreeExplainer is typically the most appropriate choice.
    explainer = shap.TreeExplainer(model)
    # Call once to ensure the background is correctly initialized
    _ = explainer(X_sample.iloc[:1])
    return explainer


def create_shap_summary_plots(shap_values, X: pd.DataFrame) -> None:
    """
    Generate and save SHAP summary plots (beeswarm + bar) for global feature importance.

    Ce plot montre l'impact global des features sur les prédictions du modèle.
    Les features en haut ont le plus d'influence. Les couleurs indiquent si la
    valeur haute/basse de la feature augmente ou diminue la prédiction
    (rouge = augmente, bleu = diminue).

    Sauvegarde du plot SHAP beeswarm montrant l'impact moyen des features
    sur les prédictions du modèle.
    """
    # Ensure output directory exists
    os.makedirs("outputs/plots", exist_ok=True)

    # Beeswarm / dot summary plot for global importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary Plot - Global Feature Importance")
    plt.tight_layout()
    plt.savefig(
        "outputs/plots/shap_summary_beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Bar plot for mean absolute SHAP value (feature importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP Summary Bar Plot - Feature Importance")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "shap_summary_bar.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        "SHAP Summary Plots (beeswarm + bar) générés et sauvegardés dans outputs/plots/"
    )


def generate_shap_individual_explanation(
    patient_idx: int,
    explainer,
    shap_values,
    X: pd.DataFrame,
    output_dir: str = "outputs/plots/",
) -> None:
    """
    Generate SHAP force and waterfall plots for a single patient's prediction.

    - Force plot: montre comment chaque feature pousse la prédiction depuis la
      valeur de base (expected value) jusqu'à la prédiction finale
      (rouge = augmente la prédiction, bleu = la diminue).
    - Waterfall plot: représente en cascade l'impact de chaque feature sur la
      prédiction, ce qui est généralement plus lisible pour les médecins.

    Utiliser cette fonction pour l'interface Streamlit/Flask afin d'expliquer
    une prédiction à un médecin pour un patient spécifique.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure we have a row for this patient
    if isinstance(X, pd.DataFrame):
        x_patient = X.iloc[patient_idx]
    else:
        x_patient = X[patient_idx]

    # --- Force plot ---
    # Note: the modern shap API often returns an Explanation object; if so,
    # shap_values[patient_idx] should already contain the base value.
    force_plot = shap.plots.force(
        explainer.expected_value,
        shap_values[patient_idx],
        x_patient,
        show=False,
    )
    # Prefer HTML export for interactive visualization
    shap.save_html(
        os.path.join(output_dir, f"force_plot_patient_{patient_idx}.html"),
        force_plot,
    )

    # --- Waterfall plot ---
    shap.plots.waterfall(shap_values[patient_idx], max_display=10, show=False)
    plt.title(f"SHAP Waterfall Plot - Patient {patient_idx}")
    plt.savefig(
        os.path.join(output_dir, f"waterfall_plot_patient_{patient_idx}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Force plot et Waterfall plot générés pour patient {patient_idx} dans {output_dir}"
    )


def convert_shap_to_ui_format(
    explainer,
    shap_values,
    X_row,
    patient_idx: int | None = None,
):
    """
    Convert SHAP outputs into a JSON‑friendly format for frontend UIs (Streamlit / Flask).

    Ce format est conçu pour être envoyé via une API ou chargé dans une app
    Streamlit/Flask. Le frontend peut ensuite afficher :
    - la base_value (valeur de base du modèle),
    - des barres colorées (rouge/bleu) pour chaque feature,
    - la somme des contributions (SHAP values) menant à la prédiction finale.
    """
    # Normalize SHAP values to a 1D array for a single instance
    if hasattr(shap_values, "values"):
        sv = shap_values.values
        base_value = getattr(shap_values, "base_values", explainer.expected_value)
    else:
        sv = np.array(shap_values)
        base_value = explainer.expected_value

    sv = np.array(sv)
    if sv.ndim > 1:
        sv_instance = sv[0]
    else:
        sv_instance = sv

    # Handle base_value potentially being an array (e.g., multiclass)
    base_value_arr = np.array(base_value)
    if base_value_arr.ndim > 0:
        base_val_scalar = float(base_value_arr[0])
    else:
        base_val_scalar = float(base_value_arr)

    # Extract feature names and values
    if isinstance(X_row, pd.Series):
        feature_names = X_row.index.tolist()
        feature_values = X_row.values
    elif isinstance(X_row, dict):
        feature_names = list(X_row.keys())
        feature_values = np.array(list(X_row.values()))
    else:
        # Fallback: use explainer.feature_names or indices
        feature_values = np.array(X_row)
        feature_names = getattr(explainer, "feature_names", None)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(feature_values))]

    # Ensure lengths match
    min_len = min(len(feature_names), len(feature_values), len(sv_instance))
    feature_names = feature_names[:min_len]
    feature_values = feature_values[:min_len]
    sv_instance = sv_instance[:min_len]

    # Sort features by absolute impact (descending)
    order = np.argsort(-np.abs(sv_instance))
    feature_names = [feature_names[i] for i in order]
    feature_values = feature_values[order]
    sv_sorted = sv_instance[order]

    features_list = []
    for name, val, sv_val in zip(feature_names, feature_values, sv_sorted):
        impact_direction = "positive" if sv_val > 0 else "negative"
        features_list.append(
            {
                "feature_name": str(name),
                "feature_value": float(val) if isinstance(val, (int, float)) else str(val),
                "shap_value": round(float(sv_val), 4),
                "impact_direction": impact_direction,
                "contribution": round(float(sv_val), 4),
            }
        )

    total_contribution = float(np.sum(sv_sorted))
    prediction = base_val_scalar + total_contribution

    # Optional force plot data for JS reconstruction (if available)
    force_plot_data = None
    try:
        fp = shap.plots.force(
            base_val_scalar,
            sv_instance,
            feature_values,
            show=False,
        )
        force_plot_data = getattr(fp, "data", None)
    except Exception:
        force_plot_data = None

    # Waterfall data: already sorted by |shap_value|
    waterfall_data = features_list

    ui_dict = {
        "prediction": round(prediction, 4),
        "base_value": round(base_val_scalar, 4),
        "features": features_list,
        "total_contribution": round(total_contribution, 4),
        "force_plot_data": force_plot_data,
        "waterfall_data": waterfall_data,
    }

    # Optionally persist to JSON for inspection / frontend loading
    if patient_idx is not None:
        os.makedirs("outputs/shap_ui", exist_ok=True)
        json_path = os.path.join("outputs/shap_ui", f"patient_{patient_idx}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(ui_dict, f, ensure_ascii=False, indent=2)
        print("SHAP data formaté pour UI et sauvegardé en JSON:", json_path)

    return ui_dict



def generate_shap_summary_plot(shap_values, X_sample: pd.DataFrame, output_path: Path) -> None:
    """Generate and save a SHAP summary beeswarm plot."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Main entry point to compute and save SHAP summary plot for the model."""
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    print(f"Loading data from: {DATA_PATH}")
    df = load_data(DATA_PATH)

    print("Preparing feature matrix X (dropping target column)...")
    X = prepare_features(df, TARGET_COLUMN)

    print("Sampling data for SHAP computation...")
    X_sample = sample_data(X, MAX_SAMPLE_SIZE)
    print(f"Sample size for SHAP: {len(X_sample)} rows.")

    print("Creating SHAP TreeExplainer for the multiclass classifier...")
    explainer = create_explainer(model, X_sample)

    print("Computing SHAP values on the sampled data...")
    shap_values = explainer(X_sample)

    # Generate classic beeswarm + bar SHAP summary plots for global feature importance
    create_shap_summary_plots(shap_values, X_sample)


if __name__ == "__main__":
    main()


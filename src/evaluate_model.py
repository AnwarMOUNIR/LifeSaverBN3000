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

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


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

    print(f"Generating SHAP summary beeswarm plot at: {OUTPUT_PLOT_PATH}")
    generate_shap_summary_plot(shap_values, X_sample, OUTPUT_PLOT_PATH)

    print(
        "SHAP summary plot généré et sauvegardé pour interprétation globale des features : "
        f"{OUTPUT_PLOT_PATH}"
    )


if __name__ == "__main__":
    main()


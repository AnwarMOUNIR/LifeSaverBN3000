import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

def calculate_metrics(y_true, y_pred, y_prob, model_name="Model"):
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except Exception:
        auc = 0.0
    return {
        "Model": model_name,
        "ROC-AUC": round(auc, 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
    }

def apply_sanity_guards(X, pred_labels):
    """
    Rule-based overrides for extreme physiologically implausible inputs.
    """
    if isinstance(pred_labels, str):
        pred_labels = [pred_labels]
        
    final_preds = []
    for idx, (_, row) in enumerate(X.iterrows()):
        h = row['Height']
        w = row['Weight']
        bmi = w / (h ** 2)
        original_pred = pred_labels[idx]
        
        # Rule 1: Very low BMI -> Must be Insufficient_Weight
        if bmi < 18.2:
            final_pred = 'Insufficient_Weight'
        # Rule 2: High BMI -> Should not be Normal or Underweight
        elif bmi > 30 and original_pred in ['Normal_Weight', 'Insufficient_Weight']:
            final_pred = 'Obesity_Type_I'
        else:
            final_pred = original_pred
        final_preds.append(final_pred)
    
    return final_preds[0] if len(final_preds) == 1 else final_preds

def run_pipeline(raw_data_path):
    print("⚡ Starting Standardized Pipeline (training on raw features)...")
    if not os.path.exists(raw_data_path):
        print(f"Error: {raw_data_path} not found.")
        return
    
    # Load raw data
    df = pd.read_csv(raw_data_path)
    X = df.iloc[:, :-1]
    y_raw = df.iloc[:, -1]
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(le, 'models/label_encoder.pkl')
    
    # Identify numeric and categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    print(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standard Scikit-Learn Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    print("🏗️ Training Random Forest...")
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', base_model)
    ])
    
    # Ensure the pipeline outputs DataFrames (essential for SHAP and App diagnostics)
    pipeline.set_output(transform="pandas")
    
    pipeline.fit(X_train, y_train)
    
    # Quick Eval
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)
    metrics = calculate_metrics(y_test, preds, probs, "RandomForest")
    print(f"✅ Training Complete. Accuracy: {metrics['Accuracy']}")
    
    # Save the pipeline
    joblib.dump(pipeline, "models/best_model.pkl")
    print("🚀 Model saved to models/best_model.pkl")

if __name__ == "__main__":
    run_pipeline('data/raw/ObesityDataSet_synthetic.csv')

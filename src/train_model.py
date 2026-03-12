import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder

# Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from src.data_processing import engineer_features

def calculate_metrics(y_true, y_pred, y_prob, model_name="Model"):
    """
    Calculates standard binary/multiclass classification metrics and returns them.
    """
    try:
        # For multiclass, we use 'ovr' (One-vs-Rest)
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

def encode_wrapper(df, reference_columns=None):
    """
    Module-level function for pickling. Uses reference_columns 
    if provided to ensure equal shapes during predict.
    """
    encoded = pd.get_dummies(df, drop_first=True)
    if reference_columns is not None:
        encoded = encoded.reindex(columns=reference_columns, fill_value=0)
    return encoded

def get_encoder(reference_columns=None):
    return FunctionTransformer(encode_wrapper, kw_args={"reference_columns": reference_columns}, validate=False)

def validate_shape(df, expected_cols=None):
    if expected_cols is not None and len(df.columns) != expected_cols:
        raise ValueError(f"Expected {expected_cols} features, got {len(df.columns)}")
    return df

def get_shape_validator(expected_cols):
    return FunctionTransformer(validate_shape, kw_args={"expected_cols": expected_cols}, validate=False)

def get_engineer_transformer():
    return FunctionTransformer(engineer_features, validate=False)

def train_model(X, y):
    """
    Trains the best model pipeline logic. Just a helper for the tests.
    """
    # Create the pipeline with a custom step to strictly check raw columns shape
    shape_validator = get_shape_validator(len(X.columns))
    
    # First get reference columns from training data
    ref_cols = pd.get_dummies(X, drop_first=True).columns
    encoder = get_encoder(ref_cols)
    model = LGBMClassifier(n_estimators=100, random_state=42)
    
    # Use an sklearn Pipeline so predict() handles encoding natively.
    pipe = Pipeline([('shape_check', shape_validator), ('encoder', encoder), ('model', model)])
    pipe.fit(X, y)
    
    return pipe

def save_model(model, path):
    """
    Saves a trained model to the specified path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    """
    Loads a trained model from the specified path.
    """
    if os.path.exists(path):
        return joblib.load(path)
    else:
        print(f"Error: Model not found at {path}")
        return None

def run_pipeline(processed_data_path):
    # 1. Load the PREPROCESSED Data
    if not os.path.exists(processed_data_path):
        print(f"Error: {processed_data_path} not found. Run data_processing.py first!")
        return
    
    df = pd.read_csv(processed_data_path)

    # 1. Separate Features (X) and Target (y)
    X = df.iloc[:, :-1] 
    y_raw = df.iloc[:, -1]
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Save label encoder for frontend to decode predictions
    os.makedirs('models', exist_ok=True)
    joblib.dump(le, 'models/label_encoder.pkl')

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Model Comparison (Wissal's valuable addition)
    print("🚀 Comparing multiple models...")
    base_models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
    }

    results = []
    print(f"{'Model':<20} | {'AUC':<8} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
    print("-" * 75)

    for name, model in base_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        
        metrics = calculate_metrics(y_test, preds, probs, model_name=name)
        results.append(metrics)
        
        print(f"{name:<20} | {metrics['ROC-AUC']:<8} | {metrics['Accuracy']:<8} | "
              f"{metrics['Precision']:<8} | {metrics['Recall']:<8} | {metrics['F1-Score']:<8}")

    # 4. Identify the Best Model
    best_model_info = max(results, key=lambda x: (x['ROC-AUC'], x['Accuracy']))
    best_model_name = best_model_info['Model']
    
    print("\n" + "=" * 75)
    print(f"🏆 WINNER: {best_model_name}. Starting Hyperparameter Tuning...")
    
    # Define the parameter grids
    param_grids = {
        "Random Forest": {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        "XGBoost": {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2]
        },
        "LightGBM": {
            'n_estimators': [100, 200],
            'max_depth': [-1, 20],
            'learning_rate': [0.1, 0.2]
        },
        "CatBoost": {
            'iterations': [100, 200],
            'depth': [4, 6],
            'learning_rate': [0.1, 0.2]
        }
    }

    best_base_model = base_models[best_model_name]
    grid = param_grids[best_model_name]

    # Set up the Random Search
    random_search = RandomizedSearchCV(
        estimator=best_base_model,
        param_distributions=grid,
        n_iter=5,
        scoring='accuracy',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    tuned_model = random_search.best_estimator_
    
    print(f"Best Params: {random_search.best_params_}")

    # 5. Final Assembly with Pipeline
    # We must save the full pipeline (encoding + model) for it to work in the app
    ref_cols = pd.get_dummies(X, drop_first=True).columns
    shape_validator = get_shape_validator(len(X.columns))
    encoder = get_encoder(ref_cols)
    
    final_pipeline = Pipeline([
        ('shape_check', shape_validator),
        ('encoder', encoder),
        ('model', tuned_model)
    ])
    
    # Final fit on full dataset or just keep the tuned version
    final_pipeline.fit(X_train, y_train) 

    # Final Save
    save_model(final_pipeline, "models/best_model.pkl")
    print("\n✔ Success! The 'Ultimate Pipeline' is ready and pushed to models/best_model.pkl")

def apply_sanity_guards(X, pred_labels):
    """
    Applies medically sound rules to override ML predictions for extreme edge cases.
    """
    if isinstance(pred_labels, str):
        pred_labels = [pred_labels]
        
    final_preds = []
    # Use enumerate to avoid index mismatch with DataFrame's index
    for idx, (i, row) in enumerate(X.iterrows()):
        h = row['Height']
        w = row['Weight']
        bmi = w / (h ** 2)
        
        original_pred = pred_labels[idx]
        
        # Rule 1: Extreme Underweight
        if bmi < 18.2:
            final_pred = 'Insufficient_Weight'
        # Rule 2: High BMI fallback
        elif bmi > 30 and original_pred in ['Normal_Weight', 'Insufficient_Weight']:
            final_pred = 'Obesity_Type_I'
        else:
            final_pred = original_pred
            
        final_preds.append(final_pred)
            
    return final_preds[0] if len(final_preds) == 1 else final_preds

if __name__ == "__main__":
    run_pipeline('data/processed/processed_data.csv')

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
def load_processed_data(file_path="data/processed/processed_data.csv"):
    """
    Standard function to load the dataset after it has been cleaned.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully loaded {len(data)} rows from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} does not exist.")
        return None
def run_pipeline(processed_data_path):
    # 1. Load the PREPROCESSED Data
    if not os.path.exists(processed_data_path):
        print(f"Error: {processed_data_path} not found. Run data_processing.py first!")
        return
    
    df = pd.read_csv(processed_data_path)

    # Separate Features and Target
    # (Since the data is already cleaned, we just split it)
    X = df.iloc[:, :-1] 
    y = df.iloc[:, -1]  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Define the 4 Models
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0)
    }

    results = []

    # 3. Train and Evaluate
    print(f"{'Model':<20} | {'AUC':<8} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
    print("-" * 75)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "ROC-AUC": round(roc_auc_score(y_test, probs), 4),
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "Precision": round(precision_score(y_test, preds), 4),
            "Recall": round(recall_score(y_test, preds), 4),
            "F1-Score": round(f1_score(y_test, preds), 4)
        }
        results.append(metrics)
        
        print(f"{name:<20} | {metrics['ROC-AUC']:<8} | {metrics['Accuracy']:<8} | "
              f"{metrics['Precision']:<8} | {metrics['Recall']:<8} | {metrics['F1-Score']:<8}")

    # 4. Save the Best Model
    best_model_info = max(results, key=lambda x: (x['ROC-AUC'], x['Accuracy']))
    best_model_name = best_model_info['Model']
    
    save_path = "models/best_model.pkl"
    joblib.dump(models[best_model_name], save_path)
    
    print("-" * 75)
    print(f"DONE! Best Model: {best_model_name} saved to {save_path}")

if __name__ == "__main__":
    # Point this to the NEW file in the processed folder
    run_pipeline('data/processed/processed_data.csv')
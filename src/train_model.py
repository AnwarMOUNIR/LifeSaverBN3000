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

def run_pipeline(data_path):
    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    
    df = pd.read_csv(data_path)

    # --- NEW: Handle Text Columns ---
    # This converts words into numbers automatically
    df = pd.get_dummies(df, drop_first=True)
    # --------------------------------

    # Separate Features and Target
    # Note: After get_dummies, your target column name might have changed 
    # if it was a string. Let's assume it's the last column.
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

        # Calculate the 5 metrics required by your task
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

    # 4. Save the Best Model (based on ROC-AUC)
    best_model_info = max(results, key=lambda x: x['ROC-AUC'])
    best_model_name = best_model_info['Model']
    
    # Save as .pkl as requested
    save_path = f"models/best_model.pkl"
    joblib.dump(models[best_model_name], save_path)
    
    print("-" * 75)
    print(f"DONE! Best Model: {best_model_name} saved to {save_path}")

if __name__ == "__main__":
    # Change this line to match your actual filename
    run_pipeline('data/ObesityDataSet_raw_and_data_sinthetic.csv')
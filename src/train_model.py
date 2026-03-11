import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def calculate_metrics(y_true, y_pred, y_prob, model_name="Model"):
    """
    Calculates standard binary classification metrics and returns them as a dictionary.
    """
    return {
        "Model": model_name,
        "ROC-AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1-Score": round(f1_score(y_true, y_pred), 4)
    }

def train_model(X, y):
    pass

def save_model(model, path):
    pass

def load_model(path):
    pass

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

    # --- Inside your run_pipeline function ---

    # 1. Separate Features (X) and Target (y)
    X = df.iloc[:, :-1] 
    y = df.iloc[:, -1]

    # 2. ADD THE SPLIT HERE
    # test_size=0.2 means 20% for testing, leaving 80% for training
    X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
    )

    # 3. Then proceed to define and train your models...
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

        # --- NEW CODE FOR YOUR TASK ---
        if name == "Random Forest":
            print(f"\n[LOG] {name} Predictions (first 15): {preds[:15]}")    
        elif name == "XGBoost":  # <-- ADD THIS PART!
            print(f"\n[LOG] {name} Predictions (first 15): {preds[:15]}")         
        elif name == "LightGBM":  # <-- ADDED FOR LIGHTGBM
            print(f"\n[LOG] {name} Predictions (first 15): {preds[:15]}")
        
        metrics = calculate_metrics(y_test, preds, probs, model_name=name)
        
        results.append(metrics)
        
        print(f"{name:<20} | {metrics['ROC-AUC']:<8} | {metrics['Accuracy']:<8} | "
              f"{metrics['Precision']:<8} | {metrics['Recall']:<8} | {metrics['F1-Score']:<8}")

    # 4. Identify the Best Model
    best_model_info = max(results, key=lambda x: (x['ROC-AUC'], x['Accuracy']))
    best_model_name = best_model_info['Model']
    
    print("\n" + "=" * 75)
    print(f"🏆 WINNER: {best_model_name}. Starting Hyperparameter Tuning...")
    
    # Define the parameter grids for each model
    param_grids = {
        "Random Forest": {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        "XGBoost": {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        "LightGBM": {
            'n_estimators': [100, 200],
            'max_depth': [-1, 10, 20],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        "CatBoost": {
            'iterations': [100, 200],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    }

    # Grab the untrained base model and its matching grid
    best_base_model = models[best_model_name]
    grid = param_grids[best_model_name]

    # Set up the Random Search
    random_search = RandomizedSearchCV(
        estimator=best_base_model,
        param_distributions=grid,
        n_iter=10,        # Number of random combinations to try
        scoring='roc_auc',# Optimize for ROC-AUC
        cv=3,             # 3-fold cross-validation
        random_state=42,
        n_jobs=-1         # Use all available CPU cores
    )

    # 5. Train the tuned model
    random_search.fit(X_train, y_train)
    tuned_model = random_search.best_estimator_
    
    print(f"✅ Tuning Complete! Best Parameters: {random_search.best_params_}")

    # 6. Save the TUNED Model
    save_path = "models/best_model.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(tuned_model, save_path)
    
    print("-" * 75)
    print(f"DONE! Tuned {best_model_name} saved to {save_path}")
    
    

if __name__ == "__main__":
    # Point this to the NEW file in the processed folder
    run_pipeline('data/processed/processed_data.csv')

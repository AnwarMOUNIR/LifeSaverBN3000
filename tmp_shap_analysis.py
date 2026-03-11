import pandas as pd
import joblib
import shap
import numpy as np

# Load data and model
df = pd.read_csv('data/processed/processed_data.csv')
X = df.iloc[:, :-1]

model = joblib.load('models/best_model.pkl')

print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
X_encoded = X

shap_values = explainer.shap_values(X_encoded)

# LGBM usually returns a list of arrays for multiclass or single array for binary.
# The obesity dataset is multiclass! 
if isinstance(shap_values, list):
    print("Multiclass output detected.")
    # Calculate feature correlation with target
    # We will just look at basic Pearson correlation of features first
    corr = X.corr(numeric_only=True)
    print("\nFeature Pearson Correlation:")
    print(corr.unstack().sort_values(ascending=False).drop_duplicates().head(15))
    
    # Or correlation between Weight and Height specifically:
    print(f"\nCorrelation between Weight and Height: {X['Weight'].corr(X['Height']):.4f}")
    
else:
    print("Binary/continuous output detected.")
    
print("\nTop important features based on mean absolute SHAP over all classes:")
# Handle list for multiclass
mean_shap = np.sum([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0) if isinstance(shap_values, list) else np.abs(shap_values).mean(axis=0)
feat_imp = pd.Series(mean_shap, index=X_encoded.columns).sort_values(ascending=False)
print(feat_imp.head(10))

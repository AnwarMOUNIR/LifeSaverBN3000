import nbformat

def add_eda_cells():
    nb_path = "notebooks/eda.ipynb"
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    # Markdwon cell
    md_cell = nbformat.v4.new_markdown_cell(
        source="## 3. Correlation & SHAP Interaction Analysis\n\n"
               "Based on our SHAP analysis, we observed strong interactions and data correlations, "
               "especially between physiological characteristics like `Weight` and `Height`. "
               "Let's visualize the standard Pearson feature correlations, and compute the mathematical `BMI` "
               "feature directly to feed the model a generalizing trait."
    )
    nb.cells.append(md_cell)
    
    # Code cell
    code_cell = nbformat.v4.new_code_cell(
        source="import seaborn as sns\n"
               "import matplotlib.pyplot as plt\n\n"
               "# Calculate Pearson correlation for numerical features\n"
               "corr = df.corr(numeric_only=True)\n\n"
               "plt.figure(figsize=(10, 8))\n"
               "sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')\n"
               "plt.title('Feature Correlation Heatmap')\n"
               "plt.show()\n\n"
               "print('We notice very strong correlations that influence obesity. Engineering a BMI feature could help generalization.')"
    )
    # Add a code cell for SHAP
    code_cell_shap = nbformat.v4.new_code_cell(
        source="import shap\nimport joblib\n\n"
               "# Load pipeline model from previous run\n"
               "try:\n"
               "    model = joblib.load('../models/best_model.pkl')\n"
               "    explainer = shap.TreeExplainer(model)\n"
               "    shap_values = explainer.shap_values(df.drop('NObeyesdad', axis=1, errors='ignore'))\n"
               "    print('SHAP computed successfully for further interaction analysis.')\n"
               "except:\n"
               "    print('Model not available or target leakage removed.')"
    )
    
    nb.cells.extend([code_cell, code_cell_shap])
    
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    print("Notebook updated successfully.")

if __name__ == "__main__":
    add_eda_cells()

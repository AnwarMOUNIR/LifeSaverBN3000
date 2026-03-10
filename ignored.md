# Project LifeSaverBN3000 Task Distribution (Pending Tasks)

Based on the current project state and the newly provided Jira Backlog and Team Organization documents, here are the updated missing tasks distributed across the team:

## TM1: Project Manager & DevOps
- [ ] Compile the final Prompt Engineering Documentation (`PROMPT_ENGINEERING.md`) from all team members.
- [ ] Verify project reproducibility (ensure `requirements.txt` works with both `train_model.py` and `app.py`).
- [ ] Answer the critical README questions (dataset balance, best model, SHAP insights, prompt insights).

## TM2: Data Engineer
- [ ] Complete `notebooks/eda.ipynb`.
- [ ] Document answers in the EDA notebook regarding missing values, outliers, handling class imbalance, and feature correlations.
- [ ] Help TM4 verify that data types optimization doesn't drop any context during preprocessing.

## TM3: Machine Learning Engineer (Data Scientist)
- [ ] Open a PR for the model training pipeline if any final adjustments are needed. (Mostly Done: `train_model.py` is implemented and exports `best_model.pkl`).
- [ ] Assist TM5 with loading the pickeled model properly in the Streamlit frontend.

## TM4: Explainable AI (XAI) & Optimization Engineer
- [ ] Implement `optimize_memory(df)` in `src/data_processing.py`.
- [ ] Implement `handle_missing_values()` and `encode_categorical()` in `src/data_processing.py` to fix the failing tests.
- [ ] Demonstrate memory savings before and after optimization in the EDA notebook.
- [ ] Document how to interpret the SHAP plots for the team.

## TM5: Frontend UI/UX Engineer
- [ ] Import the saved ML model (`models/best_model.pkl`) into `app/app.py`.
- [ ] Connect the "Predict" button to the model's `.predict()` method.
- [ ] Display the prediction result (Obesity Level) clearly to the user.
- [ ] Integrate the global SHAP Summary Plot image into an "Explainability" tab.
- [ ] Integrate individual prediction SHAP plots below the user's prediction result.

## TM6: QA & Testing Engineer
- [ ] Ensure all tests in `tests/test_setup.py` pass once TM4 implements the missing functions.
- [ ] Write a test to ensure the final model file `.pkl` exists.
- [ ] Write a test to ensure the model's predict function accepts exactly the expected number of features.
- [ ] Review CI/CD pipeline runs and fix any failing test workflows.

# Prompt Engineering Documentation
**Author:** Entire Team (DevOps Manager / TM1 to Collect)
**Date:** [Insert Date]

This file tracks the prompts you used to build `LifeSaverBN3000`, the AI's responses, and how you iteratively refined them. It is a strictly graded deliverable.

## 👥 Prompts Log by Team Member

### TM1: DevOps & Project Manager 
**Objective:** Setting up GitHub Actions CI/CD Pipeline
- **Initial Prompt:** 
  > _"Write a github actions pipeline for my python project."_
- **AI Response Summary:** 
  > _[Summarize what the AI provided - e.g., a generic Python pipeline]_
- **Refinement / Follow-up Prompt:** 
  > _"Refine this to only run on pull requests to the main branch, use Python 3.9, install requirements.txt, and run pytest specifically."_
- **Final Outcome/Code Used:**
  > _[Link to .github/workflows/python-app.yml or paste snippet]_
- **Reflection:** 
  > _"The initial prompt was too vague. Specifying the exact triggers and testing commands yielded a copy-pasteable configuration."_

---

### TM2: Data Engineer
**Objective:** Memory Optimization and Data Types
- **Initial Prompt:** 
  > _"How can I reduce the memory footprint of my pandas dataframe?"_
- **AI Response Summary:** 
  > _The AI provided a script iterating through columns to downcast integer and float types to their minimum viable size._
- **Refinement / Follow-up Prompt:** 
  > _"Can you write a reusable function `optimize_memory(df)` that also checks if `object` types can be converted to `category` by checking unique value counts?"_
- **Final Outcome/Code Used:**
  > _Implemented `optimize_memory` in `src/data_processing.py` successfully reducing DF memory usage._
- **Reflection:** 
  > _The initial prompt was generic. Refining it specifically to a reusable function and explaining the condition for `object` conversion made the code drop-in ready._

---

### TM3: Machine Learning Engineer
**Objective:** Training pipelines with LightGBM and custom categories
- **Initial Prompt:**
  > _"How do I train LightGBM?"_
- **AI Response Summary:**
  > _Provided a basic `model = LGBMClassifier().fit(X, y)` script._
- **Refinement / Follow-up Prompt:**
  > _"I am getting pandas data type issues because of categorical object columns. I need to encode them using get_dummies but also make sure it works perfectly when we use model.predict(X) in our tests. Can you wrap it in an Sklearn Pipeline?"_
- **Final Outcome/Code Used:**
  > _Created `train_model` in `src/train_model.py` which returns a full Pipeline consisting of a FunctionTransformer for dummy encoding and the LightGBM classifier._
- **Reflection:**
  > _Models need to accept raw input rows natively during deployment/testing. Moving from a basic script to strict pipeline architecture resolved dtype mismatch issues._

---

### TM4: Explainable AI & Optimization Engineer
**Objective:** Generating SHAP plots for individual predictions
- **Initial Prompt:**
  > _"How do I use SHAP in streamlit?"_
- **AI Response Summary:**
  > _Provided generic code using `shap.force_plot` and `st_shap` wrapper._
- **Refinement / Follow-up Prompt:**
  > _"Streamlit sometimes has issues with JS force plots. How can I render a static matplotlib waterfall plot for an individual prediction using TreeExplainer?"_
- **Final Outcome/Code Used:**
  > _Implemented `shap.plots.waterfall(ind_exp, show=False)` in `app/app.py` passed directly to `st.pyplot()`._
- **Reflection:**
  > _AI tend to suggest interactive JS versions of SHAP plots which require third-party streamlit components. Explicitly asking for static Matplotlib versions is much safer and cleaner for deployment._

---

### TM5: Frontend UI/UX Engineer
**Objective:** Connecting the predictive model to the interface
- **Initial Prompt:**
  > _"Write a Streamlit sidebar to input patient data."_
- **AI Response Summary:**
  > _Created standard input elements._
- **Refinement / Follow-up Prompt:**
  > _"The model was trained on `pd.get_dummies(drop_first=True)`. We need a function that takes the user inputs and identically replicates the 23 feature columns required by the model before calling predict."_
- **Final Outcome/Code Used:**
  > _Created `encode_input` in `app/app.py` mapping explicit Streamlit fields to dummy values and reindexing against exactly what the model expects._
- **Reflection:**
  > _UI inputs must exactly mirror model expectations. The prompt correctly guided the AI from standard UI generation to strict data transformation logic._

---

### TM6: QA & Testing Engineer
**Objective:** Serialization issues in tests
- **Initial Prompt:**
  > _"My joblib.dump is failing with _pickle.PicklingError: Can't pickle local object."_
- **AI Response Summary:**
  > _Explained that nested/local functions cannot be pickled by Python's standard libraries._
- **Refinement / Follow-up Prompt:**
  > _"My Local object is `def encode(df)` inside `train_model()`. How do I extract this to be compatible with Sklearn's FunctionTransformer across modules?"_
- **Final Outcome/Code Used:**
  > _Moved `encode_wrapper` and `validate_shape` outside to module-level scope in `src/train_model.py`._
- **Reflection:**
  > _Asking directly about the specific error trace helped avoid general testing theory and moved directly to a functional python syntax fix._

# LifeSaverBN3000 — Medical Obesity Risk Estimation

**LifeSaverBN3000** is a clinical decision support system designed to estimate obesity risk with high precision and transparency. Built using the UCI Obesity Dataset, it combines a robust Machine Learning pipeline with medially-grounded "Safety Guards" and Explainable AI (XAI) to ensure predictions are both accurate and interpretable by healthcare professionals.

---

## 🔬 Core Objectives
- **Precision**: Deliver 94.8%+ accuracy in classifying obesity levels across 7 categories.
- **Safety**: Eliminate model "hallucinations" (physiologically implausible predictions) using rule-based medical overrides.
- **Transparency**: Leverage SHAP (SHapley Additive exPlanations) to show exactly which lifestyle factors contribute to a patient's risk.
- **Stability**: Standardized production-ready pipeline for zero-config deployment on Streamlit Cloud.

---

## 🏗️ Architecture & Technical Design

### 1. Standardized ML Pipeline
The project uses a scikit-learn `Pipeline` to ensure consistency between training and real-time dashboard predictions.
- **Preprocessing**: Uses `ColumnTransformer` with `OneHotEncoder` (handle_unknown='ignore') and `remainder='passthrough'`.
- **Feature Engineering**: Features are kept raw to align with physician input; encoding and scaling are handled internally within the serialized `best_model.pkl`.
- **Estimator**: Standardized on a calibrated **Random Forest Classifier** for its optimal balance of predictive power and stability.

### 2. Medically-Grounded Safety Guards
To prevent AI hallucinations (e.g., predicting an underweight patient as obese), we implemented the `apply_sanity_guards` layer:
- **Underweight Guard**: Any patient with a **BMI < 18.2** is automatically flagged as `Insufficient_Weight`.
- **Overweight Guard**: Any patient with a **BMI > 30** is prevented from being classified as `Normal_Weight` or `Insufficient_Weight`.
- **Visual Feedback**: The dashboard displays a 🛡️ icon whenever a model prediction is overridden by these safety rules.

### 3. Explainable AI (XAI)
- **Individual Waterfall Plots**: Detailed breakdown of how a specific patient's habits (e.g., tech usage, water intake) push them toward a risk category.
- **Global Summary Plots**: Definitive ranking of the most influential factors across the entire dataset (Weight, Height, and Family History are top predictors).

---

## 📂 Project Structure
```text
LifeSaverBN3000/
├── .github/workflows/   # CI/CD: Automated testing on every push
├── app/                 # Dashboard: Streamlit UI, SHAP logic, and Path handling
├── data/                # Data: Raw (CSV) and insights/EDA artifacts
├── models/              # Artifacts: Serialized Pipeline (.pkl) and Label Encoder
├── src/                 # Logic: Training pipeline, metrics, and safety guards
├── tests/               # Quality: 100% pass-rate suite for training & robustness
├── Dockerfile           # Container configuration
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation (this file)
```

---

## 🚀 Installation & Local Development

### Setup Environment
1. **Clone the repository**:
   ```bash
   git clone https://github.com/AnwarMOUNIR/LifeSaverBN3000.git
   cd LifeSaverBN3000
   ```
2. **Install Dependencies**:
   For the web dashboard only:
   ```bash
   pip install -r requirements.txt
   ```
   For development (training, testing, EDA):
   ```bash
   pip install -r requirements-dev.txt
   ```

### Execution Commands
- **Retrain the Pipeline**:
  ```bash
  PYTHONPATH=. python src/train_model.py
  ```
- **Run the Dashboard**:
  ```bash
  streamlit run app/app.py
  ```
- **Execute Tests**:
  ```bash
  PYTHONPATH=. pytest tests/
  ```

---

## 🌐 Deployment & CI/CD
### Streamlit Cloud
The application is optimized for **Streamlit Cloud**. We implemented a custom `sys.path` injection in `app.py` to ensure local modules from `/src` are correctly discovered in the cloud container.

### Dockerized Deployment
Alternatively, you can run the application using **Docker**. This ensures a consistent environment across different machines. The Docker image uses a minimal set of dependencies to keep the build fast and the image small.

1. **Build the Docker Image**:
   ```bash
   docker build -t lifesaver-bn3000 .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 8501:8501 lifesaver-bn3000
   ```

3. **Access the Dashboard**:
   Open your browser and navigate to [http://localhost:8501](http://localhost:8501).

### GitHub Actions
A robust CI/CD pipeline runs on every commit, validating:
- **Model Training**: Pipeline integrity and artifact creation.
- **Robustness**: Verifying that Safety Guards correctly override edge-case inputs.
- **Data Integrity**: Memory optimization and missing value handling.

---

## 👥 Contributors & Collaboration
This project was developed during "Coding Week" with a focus on merging contributions from multiple branches into a single, high-performance production branch. Ethical considerations around synthetic data and medical oversight were prioritized during the design of the safety guard layer.

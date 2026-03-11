# LifeSaverBN3000

## 🩺 Project Description
**LifeSaverBN3000** is a Medical Decision Support Application designed for Obesity Risk Estimation with Explainable Machine Learning (XAI). This project leverages machine learning to predict obesity levels based on patient lifestyle and physical data (from the UCI Obesity Dataset) while providing clear, interpretable explanations for predictions using SHAP (SHapley Additive exPlanations) values. The end goal is to deliver an intuitive web interface for healthcare professionals to assess individual patient risks and understand the leading factors behind each prediction.

## ✨ Key Features
- **Exploratory Data Analysis (EDA) & Data Pipeline:** Pre-processing module handling missing values, numerical scaling, and class imbalances.
- **Machine Learning Models:** Training and optimization pipelines for models such as Random Forest, XGBoost, LightGBM, and CatBoost.
- **Explainable AI (XAI):** Integrated SHAP to provide global interpretability and individual patient prediction transparency.
- **Memory Optimization:** Engineered for efficiency by systematically downcasting data types for optimal memory performance.
- **Interactive Web Interface:** A Streamlit frontend providing seamless physician data input alongside real-time predictions and visual XAI insights.
- **Robust QA & Automation:** Comprehensive automated testing with `pytest` integrated within a full CI/CD GitHub Actions pipeline.

## 📊 Project Insights (TM1 DevOps/PM Report)
- **Dataset Balance:** The dataset has multiple classes for obesity levels with some synthetic data generation. The class distribution is relatively balanced except for 'Insufficient_Weight' which has fewer samples.
- **Best Model:** `Random Forest` (RandomForestClassifier) achieving high stability and 95.5% accuracy on raw baseline features. Pipeline assets saved in `models/best_model.pkl`.
- **SHAP Insights:** Global SHAP plots show that `Weight`, `Height` and `Age` are the most significant continuous features for obesity prediction. Family history also acts as a strong categorical predictor.
- **Prompt Engineering Insights:** Proper prompt context formulation heavily impacts the utility of LLM outputs. Specifying input variables and environment boundaries resulted in zero-shot workable CI/CD configurations compared to vague, open-ended "write me a pipeline" requests.

## 📂 Project Structure
```text
LifeSaverBN3000/
│
├── .github/workflows/    # CI/CD pipelines (e.g., Python app testing)
├── app/                  # Frontend Streamlit application
├── data/                 # Raw and processed datasets (UCI Obesity Dataset)
├── models/               # Saved trained ML models (.pkl files)
├── notebooks/            # Jupyter notebooks for Exploratory Data Analysis (EDA)
├── outputs/              # Saved plots and visual outputs (e.g., SHAP charts)
├── src/                  # Core ML pipeline, data processing, and tools
├── tests/                # Automated tests suite (pytest)
├── Dockerfile            # Container configuration
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation (this file)
```

## 🚀 Getting Started

### Prerequisites
- Python (Recommended: 3.9+)
- Git
- Docker (optional, for containerized run)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd LifeSaverBN3000
   ```
2. Create and activate a Virtual Environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App Locally
Run the Streamlit frontend interface:
```bash
streamlit run app/app.py
```

### Running Tests
Execute the predefined automated test suite:
```bash
pytest
```

## 🌐 Deployment
For detailed instructions on deploying to Streamlit Cloud, Docker, or Hugging Face, please refer to the [Deployment Guide](deployment_guide.md).

## 👥 Team & Collaboration
This project is built collaboratively by a cross-functional 6-member team following Agile methodologies managed via Jira, comprising DevOps, Data Engineering, ML, XAI, Frontend UI/UX, and QA Testing. For more details on team organization and responsibilities, please refer to the `ignored/team_organization.md` resource file.


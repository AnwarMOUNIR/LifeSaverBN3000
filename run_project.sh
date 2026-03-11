#!/bin/bash

echo "🩺 Starting LifeSaverBN3000 Setup..."

# Ensure we are in the right directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 1. Check if virtual environment exists, if not create it
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# 2. Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# 3. Install dependencies
echo "📥 Installing requirements..."
pip install -r requirements.txt --quiet

# 4. Ensure models and processed data exist
if [ ! -f "models/best_model.pkl" ] || [ ! -f "data/processed/processed_data.csv" ]; then
    echo "⚙️ Running data processing and training the model because files are missing..."
    python src/data_processing.py
    python src/train_model.py
else
    echo "✅ Model and processed data already exist!"
fi

# 5. Run the tests to verify everything is working
echo "🧪 Running unit tests..."
pytest tests/

# 6. Launch the Streamlit App
echo "🚀 Launching the Streamlit Web Application..."
streamlit run app/app.py

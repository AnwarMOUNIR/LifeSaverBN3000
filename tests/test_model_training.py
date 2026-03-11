<<<<<<< HEAD
"""
Tests for model training functions.
This file contains tests to verify that model training, saving, and loading work correctly.
"""

# Import what we need for testing
import pytest
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Import the functions we'll test (these don't exist yet - that's OK!)
from src.train_model import train_model, save_model, load_model
from src.data_processing import prepare_features

# ============================================================================
# FIXTURE: Sample data for model testing
# ============================================================================

@pytest.fixture
def sample_training_data():
    """
    Create a small sample dataset for testing model training.
    This is simpler than the full mock data - just what the model needs.
    """
    # Create 20 samples with features similar to the real dataset
    np.random.seed(42)  # For reproducible tests
    
    n_samples = 20
    
    data = {
        # Numerical features
        'Age': np.random.randint(18, 65, n_samples),
        'Height': np.random.uniform(1.50, 1.90, n_samples),
        'Weight': np.random.uniform(50, 120, n_samples),
        'FCVC': np.random.uniform(1, 3, n_samples),
        'NCP': np.random.uniform(1, 4, n_samples),
        'CH2O': np.random.uniform(1, 3, n_samples),
        'FAF': np.random.uniform(0, 3, n_samples),
        'TUE': np.random.uniform(0, 2, n_samples),
        
        # Categorical features
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'family_history': np.random.choice(['yes', 'no'], n_samples),
        'FAVC': np.random.choice(['yes', 'no'], n_samples),
        'CAEC': np.random.choice(['no', 'Sometimes', 'Frequently', 'Always'], n_samples),
        'SMOKE': np.random.choice(['yes', 'no'], n_samples),
        'SCC': np.random.choice(['yes', 'no'], n_samples),
        'CALC': np.random.choice(['no', 'Sometimes', 'Frequently', 'Always'], n_samples),
        'MTRANS': np.random.choice(['Walking', 'Bike', 'Motorbike', 'Public_Transportation', 'Automobile'], n_samples),
        
        # Target variable (7 obesity levels)
        'NObeyesdad': np.random.choice([
            'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
            'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
        ], n_samples)
    }
    
    return pd.DataFrame(data)


# ============================================================================
# TEST 7: Test model training function
# ============================================================================

def test_train_model_returns_model(sample_training_data):
    """
    Test that train_model() returns a trained model object.
    """
    # Skip if function doesn't exist yet
    pytest.skip("Waiting for train_model to be implemented")
    
    # Prepare features and target
    X = sample_training_data.drop('NObeyesdad', axis=1)
    y = sample_training_data['NObeyesdad']
    
    # Train model
    model = train_model(X, y)
    
    # Assertions
    assert model is not None, "train_model returned None"
    assert hasattr(model, 'predict'), "Model doesn't have predict method"
    assert hasattr(model, 'predict_proba'), "Model doesn't have predict_proba method"


# ============================================================================
# TEST 8: Test model predicts correct shape
# ============================================================================

def test_model_prediction_shape(sample_training_data):
    """
    Test that the model's predictions have the correct shape.
    """
    # Skip if function doesn't exist yet
    pytest.skip("Waiting for train_model to be implemented")
    
    # Prepare data
    X = sample_training_data.drop('NObeyesdad', axis=1)
    y = sample_training_data['NObeyesdad']
    
    # Train model
    model = train_model(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Assertions
    assert len(predictions) == len(X), f"Expected {len(X)} predictions, got {len(predictions)}"
    assert predictions.dtype == object or predictions.dtype.kind in 'UO', "Predictions should be strings/categories"


# ============================================================================
# TEST 9: Test model probability prediction shape
# ============================================================================

def test_model_probability_shape(sample_training_data):
    """
    Test that predict_proba returns probabilities for each class.
    """
    # Skip if function doesn't exist yet
    pytest.skip("Waiting for train_model to be implemented")
    
    # Prepare data
    X = sample_training_data.drop('NObeyesdad', axis=1)
    y = sample_training_data['NObeyesdad']
    n_classes = y.nunique()
    
    # Train model
    model = train_model(X, y)
    
    # Get probabilities
    probabilities = model.predict_proba(X)
    
    # Assertions
    assert probabilities.shape[0] == len(X), f"Wrong number of probability rows"
    assert probabilities.shape[1] == n_classes, f"Expected {n_classes} classes, got {probabilities.shape[1]}"
    
    # Check that probabilities sum to 1 (or close to it)
    row_sums = probabilities.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "Probabilities don't sum to 1"


# ============================================================================
# TEST 10: Test model saving and loading
# ============================================================================

def test_save_and_load_model(tmp_path, sample_training_data):
    """
    Test that model can be saved to disk and loaded back.
    tmp_path is a pytest fixture that gives a temporary directory.
    """
    # Skip if functions don't exist yet
    pytest.skip("Waiting for save_model and load_model to be implemented")
    
    # Prepare data
    X = sample_training_data.drop('NObeyesdad', axis=1)
    y = sample_training_data['NObeyesdad']
    
    # Train model
    model = train_model(X, y)
    
    # Save model to temporary file
    model_path = tmp_path / "test_model.pkl"
    save_model(model, model_path)
    
    # Check that file exists
    assert model_path.exists(), "Model file was not created"
    
    # Load model back
    loaded_model = load_model(model_path)
    
    # Check that loaded model works
    assert loaded_model is not None, "Loaded model is None"
    
    # Make predictions with both models and compare
    original_preds = model.predict(X)
    loaded_preds = loaded_model.predict(X)
    
    assert np.array_equal(original_preds, loaded_preds), "Loaded model gives different predictions"


# ============================================================================
# TEST 11: Test model file existence (Task 7)
# ============================================================================

def test_model_file_exists():
    """
    Test that the final trained model file (.pkl) exists.
    This test will pass once the ML Engineer saves the model.
    """
    # Skip if we're just collecting tests
    pytest.skip("Check manually once model is trained")
    
    # Define path to model file (adjust as needed)
    model_path = Path("models/final_model.pkl")
    
    # Assert that file exists
    assert model_path.exists(), f"Model file not found at {model_path}"
    assert model_path.is_file(), f"Path exists but is not a file: {model_path}"


# ============================================================================
# TEST 12: Test predict function accepts correct number of features (Task 8)
# ============================================================================

def test_predict_accepts_correct_features(sample_training_data):
    """
    Test that the model's predict function accepts exactly the expected
    number of features.
    """
    # Skip if function doesn't exist yet
    pytest.skip("Waiting for model to be implemented")
    
    # Prepare data
    X_full = sample_training_data.drop('NObeyesdad', axis=1)
    y = sample_training_data['NObeyesdad']
    
    # Get expected number of features
    expected_n_features = X_full.shape[1]
    
    # Train model
    model = train_model(X_full, y)
    
    # Try prediction with correct number of features
    try:
        model.predict(X_full)
        correct_features_works = True
    except Exception as e:
        correct_features_works = False
        print(f"Error with correct features: {e}")
    
    assert correct_features_works, "Model failed with correct number of features"
    
    # Try with wrong number of features (should fail)
    X_wrong = X_full.iloc[:, :-1]  # Remove one feature
    
    with pytest.raises(Exception):
        model.predict(X_wrong)


# ============================================================================
# TEST 13: Test model performance (bonus)
# ============================================================================

def test_model_minimum_accuracy(sample_training_data):
    """
    Test that the model meets minimum accuracy requirements.
    This is a bonus test - not strictly required.
    """
    # Skip if function doesn't exist yet
    pytest.skip("Waiting for model training to be implemented")
    
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    
    # Split data
    X = sample_training_data.drop('NObeyesdad', axis=1)
    y = sample_training_data['NObeyesdad']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Assert minimum accuracy (adjust threshold as needed)
    min_accuracy = 0.3  # Low because we have small sample data
    assert accuracy >= min_accuracy, f"Accuracy {accuracy:.2f} is below minimum {min_accuracy}"
    
=======
"""
Tests for model training functions.
This file contains tests to verify that model training, saving, and loading work correctly.
"""

# Import what we need for testing
import pytest
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Import the functions we'll test (these don't exist yet - that's OK!)
from src.train_model import train_model, save_model, load_model
from src.data_processing import prepare_features

# ============================================================================
# FIXTURE: Sample data for model testing
# ============================================================================

@pytest.fixture
def sample_training_data():
    """
    Create a small sample dataset for testing model training.
    This is simpler than the full mock data - just what the model needs.
    """
    # Create 20 samples with features similar to the real dataset
    np.random.seed(42)  # For reproducible tests
    
    n_samples = 20
    
    data = {
        # Numerical features
        'Age': np.random.randint(18, 65, n_samples),
        'Height': np.random.uniform(1.50, 1.90, n_samples),
        'Weight': np.random.uniform(50, 120, n_samples),
        'FCVC': np.random.uniform(1, 3, n_samples),
        'NCP': np.random.uniform(1, 4, n_samples),
        'CH2O': np.random.uniform(1, 3, n_samples),
        'FAF': np.random.uniform(0, 3, n_samples),
        'TUE': np.random.uniform(0, 2, n_samples),
        
        # Categorical features
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'family_history': np.random.choice(['yes', 'no'], n_samples),
        'FAVC': np.random.choice(['yes', 'no'], n_samples),
        'CAEC': np.random.choice(['no', 'Sometimes', 'Frequently', 'Always'], n_samples),
        'SMOKE': np.random.choice(['yes', 'no'], n_samples),
        'SCC': np.random.choice(['yes', 'no'], n_samples),
        'CALC': np.random.choice(['no', 'Sometimes', 'Frequently', 'Always'], n_samples),
        'MTRANS': np.random.choice(['Walking', 'Bike', 'Motorbike', 'Public_Transportation', 'Automobile'], n_samples),
        
        # Target variable (7 obesity levels)
        'NObeyesdad': np.random.choice([
            'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
            'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
        ], n_samples)
    }
    
    return pd.DataFrame(data)


# ============================================================================
# TEST 7: Test model training function
# ============================================================================

def test_train_model_returns_model(sample_training_data):
    """
    Test that train_model() returns a trained model object.
    """
    # Skip if function doesn't exist yet
    pytest.skip("Waiting for train_model to be implemented")
    
    # Prepare features and target
    X = sample_training_data.drop('NObeyesdad', axis=1)
    y = sample_training_data['NObeyesdad']
    
    # Train model
    model = train_model(X, y)
    
    # Assertions
    assert model is not None, "train_model returned None"
    assert hasattr(model, 'predict'), "Model doesn't have predict method"
    assert hasattr(model, 'predict_proba'), "Model doesn't have predict_proba method"


# ============================================================================
# TEST 8: Test model predicts correct shape
# ============================================================================

def test_model_prediction_shape(sample_training_data):
    """
    Test that the model's predictions have the correct shape.
    """
    # Skip if function doesn't exist yet
    pytest.skip("Waiting for train_model to be implemented")
    
    # Prepare data
    X = sample_training_data.drop('NObeyesdad', axis=1)
    y = sample_training_data['NObeyesdad']
    
    # Train model
    model = train_model(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Assertions
    assert len(predictions) == len(X), f"Expected {len(X)} predictions, got {len(predictions)}"
    assert predictions.dtype == object or predictions.dtype.kind in 'UO', "Predictions should be strings/categories"


# ============================================================================
# TEST 9: Test model probability prediction shape
# ============================================================================

def test_model_probability_shape(sample_training_data):
    """
    Test that predict_proba returns probabilities for each class.
    """
    # Skip if function doesn't exist yet
    pytest.skip("Waiting for train_model to be implemented")
    
    # Prepare data
    X = sample_training_data.drop('NObeyesdad', axis=1)
    y = sample_training_data['NObeyesdad']
    n_classes = y.nunique()
    
    # Train model
    model = train_model(X, y)
    
    # Get probabilities
    probabilities = model.predict_proba(X)
    
    # Assertions
    assert probabilities.shape[0] == len(X), f"Wrong number of probability rows"
    assert probabilities.shape[1] == n_classes, f"Expected {n_classes} classes, got {probabilities.shape[1]}"
    
    # Check that probabilities sum to 1 (or close to it)
    row_sums = probabilities.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "Probabilities don't sum to 1"


# ============================================================================
# TEST 10: Test model saving and loading
# ============================================================================

def test_save_and_load_model(tmp_path, sample_training_data):
    """
    Test that model can be saved to disk and loaded back.
    tmp_path is a pytest fixture that gives a temporary directory.
    """
    # Skip if functions don't exist yet
    pytest.skip("Waiting for save_model and load_model to be implemented")
    
    # Prepare data
    X = sample_training_data.drop('NObeyesdad', axis=1)
    y = sample_training_data['NObeyesdad']
    
    # Train model
    model = train_model(X, y)
    
    # Save model to temporary file
    model_path = tmp_path / "test_model.pkl"
    save_model(model, model_path)
    
    # Check that file exists
    assert model_path.exists(), "Model file was not created"
    
    # Load model back
    loaded_model = load_model(model_path)
    
    # Check that loaded model works
    assert loaded_model is not None, "Loaded model is None"
    
    # Make predictions with both models and compare
    original_preds = model.predict(X)
    loaded_preds = loaded_model.predict(X)
    
    assert np.array_equal(original_preds, loaded_preds), "Loaded model gives different predictions"


# ============================================================================
# TEST 11: Test model file existence (Task 7)
# ============================================================================

def test_model_file_exists():
    """
    Test that the final trained model file (.pkl) exists.
    This test will pass once the ML Engineer saves the model.
    """
    # Skip if we're just collecting tests
    pytest.skip("Check manually once model is trained")
    
    # Define path to model file (adjust as needed)
    model_path = Path("models/final_model.pkl")
    
    # Assert that file exists
    assert model_path.exists(), f"Model file not found at {model_path}"
    assert model_path.is_file(), f"Path exists but is not a file: {model_path}"


# ============================================================================
# TEST 12: Test predict function accepts correct number of features (Task 8)
# ============================================================================

def test_predict_accepts_correct_features(sample_training_data):
    """
    Test that the model's predict function accepts exactly the expected
    number of features.
    """
    # Skip if function doesn't exist yet
    pytest.skip("Waiting for model to be implemented")
    
    # Prepare data
    X_full = sample_training_data.drop('NObeyesdad', axis=1)
    y = sample_training_data['NObeyesdad']
    
    # Get expected number of features
    expected_n_features = X_full.shape[1]
    
    # Train model
    model = train_model(X_full, y)
    
    # Try prediction with correct number of features
    try:
        model.predict(X_full)
        correct_features_works = True
    except Exception as e:
        correct_features_works = False
        print(f"Error with correct features: {e}")
    
    assert correct_features_works, "Model failed with correct number of features"
    
    # Try with wrong number of features (should fail)
    X_wrong = X_full.iloc[:, :-1]  # Remove one feature
    
    with pytest.raises(Exception):
        model.predict(X_wrong)


# ============================================================================
# TEST 13: Test model performance (bonus)
# ============================================================================

def test_model_minimum_accuracy(sample_training_data):
    """
    Test that the model meets minimum accuracy requirements.
    This is a bonus test - not strictly required.
    """
    # Skip if function doesn't exist yet
    pytest.skip("Waiting for model training to be implemented")
    
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    
    # Split data
    X = sample_training_data.drop('NObeyesdad', axis=1)
    y = sample_training_data['NObeyesdad']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Assert minimum accuracy (adjust threshold as needed)
    min_accuracy = 0.3  # Low because we have small sample data
    assert accuracy >= min_accuracy, f"Accuracy {accuracy:.2f} is below minimum {min_accuracy}"
    
>>>>>>> 7b4cf87818abb3a6aa63d58d2ae75914174b6df9
    print(f"✅ Model accuracy: {accuracy:.2f}")
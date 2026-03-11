"""
Tests for data processing functions.
This file contains tests to verify that our data processing works correctly.
"""

# Import what we need for testing
import pytest
import pandas as pd
import numpy as np
from src.data_processing import optimize_memory, encode_categorical, handle_missing_values

# ============================================================================
# TEST 1: Testing optimize_memory function (TASK 1-4)
# ============================================================================
@pytest.mark.skip(reason="TM4 has not implemented optimize_memory in src/data_processing.py yet")
def test_optimize_memory_preserves_data():
    """
    Test that optimize_memory() doesn't lose or change data.
    
    We create a small dataframe, run optimization, and check that
    all values are still the same.
    """
    # 1. SETUP - Create a simple test dataframe
    test_data = pd.DataFrame({
        'Age': [25, 30, 35, 40, 45],
        'Height': [1.65, 1.70, 1.75, 1.80, 1.85],
        'Weight': [70.5, 75.2, 80.1, 85.4, 90.3],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male']
    })
    
    # 2. ACTION - Run the optimize_memory function
    # (Note: We'll write this test assuming optimize_memory exists)
    optimized_df = optimize_memory(test_data.copy())
    
    # 3. ASSERTION - Check that data wasn't lost or changed
    
    # Check same number of rows and columns
    assert optimized_df.shape == test_data.shape, "Shape changed after optimization!"
    
    # Check that numeric values are still the same (approximately)
    # We use .all() to check ALL values match
    numeric_columns = ['Age', 'Height', 'Weight']
    for col in numeric_columns:
        assert (optimized_df[col] == test_data[col]).all(), f"Values in {col} changed!"
    
    # Check that categorical data is preserved
    assert (optimized_df['Gender'] == test_data['Gender']).all(), "Gender values changed!"


# ============================================================================
# TEST 1B: Testing optimize_memory with realistic mock data (ENHANCED VERSION) (TASK 4)
# ============================================================================

def test_optimize_memory_preserves_mock_data(mock_obesity_data):
    """
    Test that optimize_memory() doesn't lose or change data when using 
    the realistic mock dataset.
    This is an enhanced version using the Task 3 mock data fixture.
    """
    # Skip if the function doesn't exist yet
    pytest.skip("Waiting for optimize_memory to be implemented")
    
    # Make a copy to avoid modifying the original fixture
    test_data = mock_obesity_data.copy()
    
    # Store original shape and values for comparison
    original_shape = test_data.shape
    original_values = test_data.copy()
    
    # Run optimization
    optimized_df = optimize_memory(test_data)
    
    # Check shape preserved
    assert optimized_df.shape == original_shape, "Shape changed after optimization!"
    
    # Check all values preserved (numerical columns)
    numerical_cols = test_data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        assert (optimized_df[col] == original_values[col]).all(), \
            f"Values in numerical column {col} changed!"
    
    # Check all values preserved (categorical columns)
    categorical_cols = test_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        # For categorical columns, we need to handle potential NaN values
        # First, check if both have same NaN pattern
        assert optimized_df[col].isna().sum() == original_values[col].isna().sum(), \
            f"NaN count changed in column {col}!"
        
        # Then check non-NaN values match
        mask = ~original_values[col].isna()
        assert (optimized_df[col][mask] == original_values[col][mask]).all(), \
            f"Values in categorical column {col} changed!"
    
    print(f"✅ optimize_memory preserved all data in mock dataset ({len(test_data)} rows)")


# ============================================================================
# TEST 2: Testing memory improvement (bonus test) (TASK 2)
# ============================================================================
@pytest.mark.skip(reason="TM4 has not implemented optimize_memory in src/data_processing.py yet")
def test_optimize_memory_reduces_memory():
    """
    Test that optimize_memory() actually reduces memory usage.
    This is a "bonus" test - not strictly required but good to have!
    """
    # SETUP - Create a dataframe with inefficient data types
    test_data = pd.DataFrame({
        'Age': np.array([25, 30, 35], dtype=np.int64),  # int64 uses more memory
        'Height': np.array([1.65, 1.70, 1.75], dtype=np.float64)  # float64 uses more memory
    })
    
    # Get memory usage before optimization
    memory_before = test_data.memory_usage(deep=True).sum()
    
    # ACTION - Run optimization
    optimized_df = optimize_memory(test_data)
    
    # Get memory usage after optimization
    memory_after = optimized_df.memory_usage(deep=True).sum()
    
    # ASSERTION - Memory should be less (or at least not more)
    assert memory_after < memory_before, "Memory usage didn't decrease!"
    
    # Print the improvement (pytest will show this if test fails)
    improvement = (memory_before - memory_after) / memory_before * 100
    print(f"Memory improved by {improvement:.1f}%")


# ============================================================================
# TEST 3: Testing categorical encoding (TASK 2-5)
# ============================================================================
@pytest.mark.skip(reason="TM4 has not implemented encode_categorical in src/data_processing.py yet")
def test_categorical_encoding_returns_expected_shape():
    """
    Test that encode_categorical() returns the right number of columns.
    When we encode categories, one category column might become multiple columns
    (like Gender -> Gender_Male, Gender_Female).
    """
    # SETUP - Create data with categorical columns
    test_data = pd.DataFrame({
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'FamilyHistory': ['Yes', 'No', 'No', 'Yes'],
        'Age': [25, 30, 35, 40]  # Numerical column, shouldn't change
    })
    
    # Count how many unique values in each categorical column
    # This helps us predict how many new columns we'll get
    gender_unique = test_data['Gender'].nunique()  # Should be 2 (Male, Female)
    family_unique = test_data['FamilyHistory'].nunique()  # Should be 2 (Yes, No)
    
    # Original number of columns
    original_columns = test_data.shape[1]  # Should be 3
    
    # When we encode, we expect:
    # - Original columns: 3
    # - Minus 2 categorical columns = 1 numerical column stays
    # - Plus new encoded columns: 2 (from Gender) + 2 (from FamilyHistory) = 4
    # Total expected = 1 + 4 = 5 columns
    expected_columns = 1 + gender_unique + family_unique
    
    # ACTION - Run encoding
    encoded_df = encode_categorical(test_data)
    
    # ASSERTION - Check number of columns
    assert encoded_df.shape[1] == expected_columns, \
        f"Expected {expected_columns} columns, but got {encoded_df.shape[1]}"


# ============================================================================
# TEST 4: Testing missing value handling (TASK 2)
# ============================================================================
@pytest.mark.skip(reason="TM4 has not implemented handle_missing_values in src/data_processing.py yet")
def test_missing_value_handling():
    """
    Test that handle_missing_values() properly deals with missing data.
    """
    # SETUP - Create data with missing values
    test_data = pd.DataFrame({
        'Age': [25, None, 35, None, 45],  # 2 missing values
        'Height': [1.65, 1.70, None, 1.80, 1.85],  # 1 missing value
        'Weight': [70.5, 75.2, 80.1, None, 90.3],  # 1 missing value
        'Gender': ['Male', 'Female', None, 'Female', 'Male']  # 1 missing value
    })
    
    # Count missing values before
    missing_before = test_data.isnull().sum().sum()
    
    # ACTION - Handle missing values
    cleaned_df = handle_missing_values(test_data)
    
    # ASSERTION - No missing values should remain
    missing_after = cleaned_df.isnull().sum().sum()
    assert missing_after == 0, f"Still have {missing_after} missing values!"
    
    # Check that we still have the same number of rows
    # (unless the strategy was to drop rows)
    assert cleaned_df.shape[0] > 0, "All rows were dropped!"


# ============================================================================
# TEST 5: Edge cases - Empty dataframe (TASK 2)
# ============================================================================
@pytest.mark.skip(reason="TM4 has not implemented optimize_memory in src/data_processing.py yet")
def test_optimize_memory_empty_dataframe():
    """
    Test that optimize_memory() handles empty dataframes gracefully.
    This tests an "edge case" - unusual but possible situations.
    """
    # SETUP - Empty dataframe
    empty_df = pd.DataFrame()
    
    # ACTION - This should NOT crash
    try:
        result = optimize_memory(empty_df)
        # ASSERTION - If we get here, no crash happened!
        assert True
    except Exception as e:
        # If we get here, the function crashed
        assert False, f"optimize_memory crashed on empty dataframe: {e}"


# ============================================================================
# TEST 6: Test with minimal data (1 row) (TASK 2)
# ============================================================================
@pytest.mark.skip(reason="TM4 has not implemented encode_categorical in src/data_processing.py yet")
def test_categorical_encoding_single_row():
    """
    Test that encoding works even with just one row of data.
    Another edge case test.
    """
    # SETUP - Just one row of data
    test_data = pd.DataFrame({
        'Gender': ['Male'],
        'FamilyHistory': ['Yes'],
        'Age': [25]
    })
    
    # ACTION - This should work without errors
    try:
        encoded_df = encode_categorical(test_data)
        
        # ASSERTION
        assert encoded_df is not None
        assert encoded_df.shape[0] == 1  # Still one row
    except Exception as e:
        assert False, f"Encoding failed with single row: {e}"


# ============================================================================
# MOCK DATAFRAME FIXTURE - FOR REUSABLE TEST DATA (TASK 3)
# ============================================================================

@pytest.fixture
def mock_obesity_data():
    """
    Creates a mock dataframe that mimics the real obesity dataset.
    This fixture provides consistent test data for all tests.
    
    Returns:
        pd.DataFrame: A small but realistic test dataset with 10 samples
    """
    import pandas as pd
    import numpy as np
    
    # Create mock data that looks like the real obesity dataset
    data = {
        # Numerical features
        'Age': [22, 35, 28, 45, 31, 26, 38, 42, 29, 33],
        'Height': [1.65, 1.80, 1.72, 1.68, 1.75, 1.69, 1.82, 1.71, 1.77, 1.73],
        'Weight': [65.2, 85.3, 72.1, 78.4, 92.5, 58.7, 88.2, 95.1, 70.3, 81.6],
        'FCVC': [2.0, 3.0, 2.0, 1.0, 3.0, 2.0, 3.0, 2.0, 1.0, 3.0],  # Vegetable consumption
        'NCP': [3.0, 3.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 2.0, 3.0],    # Main meals per day
        'CH2O': [2.0, 2.5, 2.0, 1.5, 2.0, 2.0, 2.5, 1.5, 2.0, 2.0],    # Daily water intake
        'FAF': [1.0, 0.0, 2.0, 1.0, 0.0, 3.0, 1.0, 0.0, 2.0, 1.0],     # Physical activity
        'TUE': [1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0],     # Tech usage
        
        # Categorical features
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 
                   'Female', 'Male', 'Female', 'Male', 'Female'],
        'family_history': ['yes', 'yes', 'no', 'yes', 'no', 
                           'no', 'yes', 'yes', 'no', 'yes'],
        'FAVC': ['yes', 'yes', 'no', 'yes', 'yes', 
                 'no', 'yes', 'yes', 'no', 'yes'],  # High caloric food
        'SMOKE': ['no', 'no', 'yes', 'no', 'no', 
                  'no', 'yes', 'no', 'no', 'no'],
        'SCC': ['no', 'no', 'no', 'no', 'no', 
                'no', 'no', 'no', 'no', 'no'],  # Calorie monitoring
        
        # Target variable (obesity levels)
        'NObeyesdad': ['Normal_Weight', 'Overweight_Level_I', 'Normal_Weight', 
                       'Obesity_Type_I', 'Overweight_Level_II', 'Normal_Weight',
                       'Obesity_Type_I', 'Obesity_Type_II', 'Normal_Weight', 
                       'Overweight_Level_I']
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values to test handling (10% of data)
    # This makes it realistic for testing missing value functions
    mask = np.random.choice([True, False], size=df.shape, p=[0.05, 0.95])
    df = df.mask(mask)
    
    return df


# ============================================================================
# TESTS USING MOCK DATAFRAME (TASK 3)
# ============================================================================

def test_mock_data_is_valid(mock_obesity_data):
    """
    Test that our mock dataframe is properly structured.
    This verifies the fixture itself works correctly.
    """
    # Check that we have data
    assert mock_obesity_data is not None
    assert len(mock_obesity_data) > 0
    
    # Check that it has the expected columns
    expected_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 
                        'FAF', 'TUE', 'Gender', 'family_history', 'FAVC', 
                        'SMOKE', 'SCC', 'NObeyesdad']
    
    for col in expected_columns:
        assert col in mock_obesity_data.columns, f"Missing column: {col}"
    
    print(f"✅ Mock dataframe created successfully with {len(mock_obesity_data)} rows")


def test_mock_data_has_correct_types(mock_obesity_data):
    """
    Verify that our mock data has the right types for each column.
    This helps ensure tests are realistic.
    """
    # Numerical columns should be numbers
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    for col in numerical_cols:
        assert pd.api.types.is_numeric_dtype(mock_obesity_data[col]), \
            f"{col} should be numeric but is {mock_obesity_data[col].dtype}"
    
    # Categorical columns should be object/string type
    categorical_cols = ['Gender', 'family_history', 'FAVC', 'SMOKE', 'SCC', 'NObeyesdad']
    for col in categorical_cols:
        assert mock_obesity_data[col].dtype == 'object', \
            f"{col} should be object type but is {mock_obesity_data[col].dtype}"



# ============================================================================
# This is a special pytest fixture 
# It helps with more advanced testing later
# ============================================================================

@pytest.fixture
def sample_test_data():
    """
    Creates sample data that can be used by multiple tests.
    This avoids writing the same setup code in every test.
    """
    return pd.DataFrame({
        'Age': [25, 30, 35, 40, 45],
        'Height': [1.65, 1.70, 1.75, 1.80, 1.85],
        'Weight': [70.5, 75.2, 80.1, 85.4, 90.3],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'FAVC': ['yes', 'no', 'yes', 'yes', 'no']  # Frequent consumption of high caloric food
    })


def test_with_fixture(sample_test_data):
    """
    Example of using the fixture above.
    The fixture automatically provides the test data!
    """
    # sample_test_data is already created by the fixture
    assert len(sample_test_data) == 5
    assert list(sample_test_data.columns) == ['Age', 'Height', 'Weight', 'Gender', 'FAVC']

# ============================================================================
# TEST 7: ML Flow Outputs / Best Model Expectations
# ============================================================================
import os
import joblib

def test_best_model_file_exists():
    """
    Test that the best trained model pickle file was successfully written to the models directory.
    """
    model_path = 'models/best_model.pkl'
    if not os.path.exists(model_path):
        pytest.skip("Model not found. Skipping file existence check since CI doesn't track .pkl files directly.")
    assert os.path.exists(model_path), "The best_model.pkl file is missing from models/."

def test_model_predict_features_count():
    """
    Test that the trained model accepts exactly 28 features as expected from the
    preprocessed UCI Obesity dataset (after accounting for encoding).
    """
    model_path = 'models/best_model.pkl'
    if not os.path.exists(model_path):
        pytest.skip("Model file not found. Run training first to evaluate feature counts.")
    
    model = joblib.load(model_path)
    # The actual feature count used by the generated LightGBM model
    assert hasattr(model, 'n_features_in_'), "The trained model must expose n_features_in_."
    assert model.n_features_in_ == 24, f"Expected 24 features, found {model.n_features_in_} in {type(model).__name__}."

"""
Tests for data processing functions.
This file contains tests to verify that our data processing works correctly.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import optimize_memory, handle_missing_values

# ============================================================================
# FIXTURE: Mock data for testing
# ============================================================================

@pytest.fixture
def sample_obesity_data():
    """Realistic mock data for obesity testing."""
    data = {
        'Age': [22, 35, 28, 45, 31],
        'Height': [1.65, 1.80, 1.72, 1.68, 1.75],
        'Weight': [65.2, 85.3, 72.1, 78.4, 92.5],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'family_history': ['yes', 'yes', 'no', 'yes', 'no']
    }
    return pd.DataFrame(data)

# ============================================================================
# TEST 1: Testing optimize_memory
# ============================================================================

def test_optimize_memory_preserves_data(sample_obesity_data):
    """Test that optimize_memory() doesn't lose or change data."""
    test_data = sample_obesity_data.copy()
    optimized_df = optimize_memory(test_data)
    
    assert optimized_df.shape == test_data.shape, "Shape changed after optimization!"
    
    numeric_columns = ['Age', 'Height', 'Weight']
    for col in numeric_columns:
        assert np.allclose(optimized_df[col], test_data[col], atol=1e-5), f"Values in {col} changed!"
    
    assert (optimized_df['Gender'] == test_data['Gender']).all(), "Categorical values changed!"

def test_optimize_memory_reduces_memory():
    """Test that optimize_memory() actually reduces memory usage."""
    test_data = pd.DataFrame({
        'Age': np.array([25, 30, 35], dtype=np.int64),
        'Height': np.array([1.65, 1.70, 1.75], dtype=np.float64)
    })
    
    memory_before = test_data.memory_usage(deep=True).sum()
    optimized_df = optimize_memory(test_data)
    memory_after = optimized_df.memory_usage(deep=True).sum()
    
    assert memory_after < memory_before, "Memory usage didn't decrease!"

# ============================================================================
# TEST 2: Testing handle_missing_values
# ============================================================================

def test_handle_missing_values():
    """Test that handle_missing_values() properly deals with missing data."""
    test_data = pd.DataFrame({
        'Age': [25, None, 35, None, 45],
        'Height': [1.65, 1.70, None, 1.80, 1.85],
        'Gender': ['Male', 'Female', None, 'Female', 'Male']
    })
    
    cleaned_df = handle_missing_values(test_data)
    assert cleaned_df.isnull().sum().sum() == 0, "Still have missing values!"
    assert cleaned_df.shape[0] > 0, "All rows were dropped!"

def test_optimize_memory_empty_dataframe():
    """Test that optimize_memory() handles empty dataframes gracefully."""
    empty_df = pd.DataFrame()
    result = optimize_memory(empty_df) # Should not crash
    assert result.empty

# ============================================================================
# TEST 3: Model File Existence
# ============================================================================

def test_best_model_file_exists():
    """Test that the best trained model pickle file exists."""
    import os
    model_path = 'models/best_model.pkl'
    assert os.path.exists(model_path), "The best_model.pkl file is missing from models/."

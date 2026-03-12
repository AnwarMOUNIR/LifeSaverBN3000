import pytest
import pandas as pd
import numpy as np
import joblib
import os
from src.train_model import apply_sanity_guards

def test_sanity_guard_underweight():
    """
    Test that the sanity guard correctly identifies extreme underweight cases.
    Input: 25yo Male, 1.7m, 20kg -> BMI ~6.9
    """
    input_data = pd.DataFrame({
        "Gender": ["Male"], "Age": [25], "Height": [1.7], "Weight": [20],
        "family_history_with_overweight": ["no"], "FAVC": ["no"],
        "FCVC": [2.0], "NCP": [3.0], "CAEC": ["Sometimes"], "SMOKE": ["no"],
        "CH2O": [2.0], "SCC": ["no"], "FAF": [1.0], "TUE": [1.0],
        "CALC": ["no"], "MTRANS": ["Public_Transportation"]
    })
    
    model_pred = "Normal_Weight" # Simulate a "hallucination"
    
    final_pred = apply_sanity_guards(input_data, model_pred)
    
    assert final_pred == "Insufficient_Weight", f"Guard failed! Expected Insufficient_Weight, got {final_pred}"

def test_sanity_guard_obese_hallucination():
    """
    Test guard override when BMI is very high but model says Normal.
    Input: 30yo Female, 1.6m, 150kg -> BMI ~58.6
    """
    input_data = pd.DataFrame({
        "Gender": ["Female"], "Age": [30], "Height": [1.6], "Weight": [150],
        "family_history_with_overweight": ["yes"], "FAVC": ["yes"],
        "FCVC": [3.0], "NCP": [3.0], "CAEC": ["Frequently"], "SMOKE": ["no"],
        "CH2O": [3.0], "SCC": ["no"], "FAF": [0.0], "TUE": [2.0],
        "CALC": ["no"], "MTRANS": ["Automobile"]
    })
    
    model_pred = "Normal_Weight" # Simulate hallucination
    
    final_pred = apply_sanity_guards(input_data, model_pred)
    
    assert final_pred == "Obesity_Type_I", f"Guard failed! Expected Obesity_Type_I for BMI > 30, got {final_pred}"

def test_sanity_guard_normal_passes_through():
    """
    Test that valid predictions are not changed.
    """
    input_data = pd.DataFrame({
        "Gender": ["Male"], "Age": [25], "Height": [1.8], "Weight": [75],
        "family_history_with_overweight": ["no"], "FAVC": ["no"],
        "FCVC": [2.0], "NCP": [3.0], "CAEC": ["Sometimes"], "SMOKE": ["no"],
        "CH2O": [2.0], "SCC": ["no"], "FAF": [1.0], "TUE": [1.0],
        "CALC": ["no"], "MTRANS": ["Public_Transportation"]
    })
    # BMI = 75 / 1.8^2 = 23.15 (Normal)
    
    model_pred = "Normal_Weight"
    
    final_pred = apply_sanity_guards(input_data, model_pred)
    
    assert final_pred == "Normal_Weight"

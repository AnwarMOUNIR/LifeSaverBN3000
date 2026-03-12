import pandas as pd
import numpy as np
import os

def optimize_memory(df):
    """
    Optimizes memory usage by adjusting data types.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB")
    
    return df

def encode_categorical(df):
    """
    Encodes categorical features. Be sure not to encode the target variable.
    """
    target_col = 'NObeyesdad'
    if target_col in df.columns:
        target = df[target_col]
        features = df.drop(columns=[target_col])
        features_encoded = pd.get_dummies(features, drop_first=True)
        # Re-attach target
        features_encoded[target_col] = target
        return features_encoded
    return pd.get_dummies(df, drop_first=True)

def handle_missing_values(df):
    """
    Checks for missing values. As per UCI documentation, there are none, so we just verify and return.
    """
    if df.isnull().sum().sum() > 0:
        print("Warning: Missing values found, but not expected based on dataset info.")
    return df

def engineer_features(df):
    """
    (Deprecated) Generates new mathematical features based on SHAP.
    We removed BMI because it was too strongly correlated and dominated the ML model.
    """
    return df

def prepare_features(df):
    """
    Combines the steps.
    """
    df = handle_missing_values(df)
    df = engineer_features(df)
    df = optimize_memory(df)
    df = encode_categorical(df)
    return df

def process_data(input_path, output_filename="processed_data.csv"):
    # 1. Load the raw data
    df = pd.read_csv(input_path)
    
    # 2. Perform your transformations
    df_processed = prepare_features(df)
    
    # 3. Define the output path
    output_path = os.path.join("data", "processed", output_filename)
    
    # 4. Save it!
    df_processed.to_csv(output_path, index=False)
    print(f"✔ Success! Processed data saved to: {output_path}")
    
    return output_path

def load_processed_data(file_name="processed_data.csv"):
    """
    Loads the processed dataset from data/processed/.
    """
    path = os.path.join("data", "processed", file_name)
    
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"Error: {path} not found.")
        return None

if __name__ == "__main__":
    # Support both original and Wissal's synthetic dataset
    input_file = "data/raw/ObesityDataSet_synthetic.csv"
    if not os.path.exists(input_file):
        input_file = "data/ObesityDataSet_raw_and_data_sinthetic.csv"
        
    process_data(input_file)

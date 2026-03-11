import pandas as pd
import os

# TM4 TODO: Implement these functions
def optimize_memory(df):
    pass

def encode_categorical(df):
    pass

def handle_missing_values(df):
    pass

def process_data(input_path, output_filename="processed_data.csv"):
    # 1. Load the raw data
    df = pd.read_csv(input_path)
    
    # 2. Perform your transformations (like get_dummies)
    # This turns 'Gender', 'Family History', etc., into 0s and 1s
    df_processed = pd.get_dummies(df, drop_first=True)
    
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
    # Run this to test it
    process_data("data/ObesityDataSet_raw_and_data_sinthetic.csv")

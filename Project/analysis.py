import pandas as pd

df = pd.read_csv("data/ObesityDataSet_raw_and_data_sinthetic.csv")

df.info()
print(df.describe())
print(df.isnull().sum())
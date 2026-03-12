import pandas as pd

df = pd.read_csv("../data/raw/ObesityDataSet_synthetic.csv")

df.info()
print(df.describe())
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 10.1 Load dataset
df = pd.read_csv("data/ObesityDataSet_raw_and_data_sinthetic.csv")

# 10.2 Basic info
df.info()
print(df.describe())

# 11 Missing values
print(df.isnull().sum())

# 12 Handle missing values
df = df.dropna()

# 13: Histograms
df.hist(figsize=(12,10))
plt.show()

# 14 Bar charts
df['Gender'].value_counts().plot(kind='bar')
plt.title("Gender Distribution")
plt.show()

df['family_history_with_overweight'].value_counts().plot(kind='bar')
plt.title("Family History of Overweight")
plt.show()

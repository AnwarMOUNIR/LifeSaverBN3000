import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/ObesityDataSet_raw_and_data_sinthetic.csv")

df.info()
print(df.describe())
print(df.isnull().sum())

# handle missing values
df = df.dropna()

df.hist(figsize=(12,10))
plt.show()

# bar charts for categorical variables

df['Gender'].value_counts().plot(kind='bar')
plt.title("Gender Distribution")
plt.show()

df['family_history_with_overweight'].value_counts().plot(kind='bar')
plt.title("Family History of Overweight")
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create insights directory if it doesn't exist
os.makedirs("data/insights", exist_ok=True)

# Support both original and stabilized path
input_path = "data/raw/ObesityDataSet_synthetic.csv"
if not os.path.exists(input_path):
    input_path = "data/ObesityDataSet_raw_and_data_sinthetic.csv"

# 10.1 Load dataset
df = pd.read_csv(input_path)

# 10.2 Basic info
print("--- Dataset Info ---")
df.info()
print("\n--- Descriptive Statistics ---")
print(df.describe())

# 11 Missing values
print("\n--- Missing Values ---")

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
plt.tight_layout()
plt.savefig("data/insights/histograms.png")
print("Saved data/insights/histograms.png")
plt.close()

# 14 Bar charts
plt.figure(figsize=(8,6))
df['Gender'].value_counts().plot(kind='bar')
plt.title("Gender Distribution")
plt.savefig("data/insights/gender_distribution.png")
print("Saved data/insights/gender_distribution.png")
plt.close()

plt.figure(figsize=(8,6))
df['family_history_with_overweight'].value_counts().plot(kind='bar')
plt.title("Family History of Overweight")
plt.savefig("data/insights/family_history.png")
print("Saved data/insights/family_history.png")
plt.close()

# 15 Correlation heatmap
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.show()

# 14 Bar charts
df['Gender'].value_counts().plot(kind='bar')
plt.title("Gender Distribution")
plt.show()

df['family_history_with_overweight'].value_counts().plot(kind='bar')
plt.title("Family History of Overweight")
plt.show()

# 15 Correlation heatmap
numeric_df = df.select_dtypes(include=['float64', 'int64'])

corr = numeric_df.corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("data/insights/correlation_heatmap.png")
print("Saved data/insights/correlation_heatmap.png")
plt.close()

plt.title("Correlation Heatmap")
plt.show()
# 16: Class distribution of obesity levels
df['NObeyesdad'].value_counts().plot(kind='bar')

plt.title("Distribution of Obesity Levels")
plt.xlabel("Obesity Level")
plt.ylabel("Count")

plt.show()

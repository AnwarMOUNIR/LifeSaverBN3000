import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create insights directory if it doesn't exist
os.makedirs("data/insights", exist_ok=True)

# Stabilized raw data path
input_path = "data/raw/ObesityDataSet_synthetic.csv"

# 10.1 Load dataset
df = pd.read_csv(input_path)

# 10.2 Basic info
print("--- Dataset Info ---")
df.info()
print("\n--- Descriptive Statistics ---")
print(df.describe())

# 11 Missing values
print("\n--- Missing Values ---")
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

# 16 BMI Distribution and Sanity Check
df['BMI'] = df['Weight'] / (df['Height'] ** 2)
plt.figure(figsize=(10,6))
sns.histplot(df['BMI'], kde=True)
plt.axvline(18.5, color='r', linestyle='--', label='Underweight (<18.5)')
plt.axvline(25, color='g', linestyle='--', label='Overweight (>25)')
plt.axvline(30, color='orange', linestyle='--', label='Obese (>30)')
plt.title("Distribution of BMI in Dataset")
plt.legend()
plt.savefig("data/insights/bmi_distribution.png")
print("Saved data/insights/bmi_distribution.png")
plt.close()

# 17: Class distribution of obesity levels
plt.figure(figsize=(10,6))
df['NObeyesdad'].value_counts().plot(kind='bar')
plt.title("Distribution of Obesity Levels")
plt.xlabel("Obesity Level")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/insights/class_distribution.png")
print("Saved data/insights/class_distribution.png")
plt.close()

# Identify potential "blind spots" (Underweight where BMI is extremely low)
blind_spots = df[(df['BMI'] < 16) & (df['NObeyesdad'] != 'Insufficient_Weight')]
if not blind_spots.empty:
    print(f"\n[ALERT] Found {len(blind_spots)} records with BMI < 16 not labeled as Insufficient Weight.")
else:
    print("\n[INFO] No logical inconsistencies found in current training labels for BMI < 16.")

# 18 Correlation heatmap
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("data/insights/correlation_heatmap.png")
print("Saved data/insights/correlation_heatmap.png")
plt.close()

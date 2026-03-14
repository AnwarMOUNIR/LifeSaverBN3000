import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
raw_data_path = "data/raw/ObesityDataSet_synthetic.csv"
df = pd.read_csv(raw_data_path)

# Calculate BMI
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

# Plotting
plt.figure(figsize=(10, 6))
sns.histplot(df['BMI'], kde=True, color='skyblue', bins=30)

# Highlight Safety Guard Zone
plt.axvspan(0, 18.2, color='red', alpha=0.2, label='Safety Guard Zone (Low Data)')
plt.axvline(18.2, color='red', linestyle='--', linewidth=2)

# Annotations
plt.text(14, 150, "Sparse Training Data\nin this region", color='red', fontweight='bold', ha='center')
plt.annotate('Rule: BMI < 18.2 → Insufficient', xy=(18.2, 200), xytext=(22, 250),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=10, fontweight='bold')

plt.title("Rationale for Safety Guards: Monitoring Sparse Regions", fontsize=14)
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.legend()

# Save
output_path = "data/insights/safety_guard_rationale.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight', dpi=150)
print(f"Visualization saved to {output_path}")

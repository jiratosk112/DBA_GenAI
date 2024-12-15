import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config

# Load the dataset
data = pd.read_csv(config.DATA_FILE)

# Step 1: Data Overview
print("Dataset Info:")
data.info()
print("\nMissing Values:")
print(data.isnull().sum())
print("\nSummary Statistics:")
print(data.describe())

# Step 2: Data Cleaning
# Convert TotalCharges to numeric and handle errors
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Drop rows with missing TotalCharges
data_cleaned = data.dropna(subset=['TotalCharges'])

# Verify cleaning
print(f"\nRows after cleaning: {data_cleaned.shape[0]} (out of {data.shape[0]})")

# Step 3: Univariate Analysis
# Numerical Features
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
for feature in numerical_features:
    plt.figure(figsize=(8, 5))
    plt.hist(data_cleaned[feature], bins=30, edgecolor='k', alpha=0.7)
    plt.title(f'Distribution of {feature}', fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.show()

# Categorical Features: Churn Counts
plt.figure(figsize=(8, 5))
data_cleaned['Churn'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'], edgecolor='k', alpha=0.7)
plt.title('Churn Counts', fontsize=14)
plt.xlabel('Churn', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=0)
plt.show()

# Step 4: Bivariate Analysis
# Tenure vs. Churn
plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='tenure', data=data_cleaned, palette='Set2')
plt.title('Tenure vs. Churn', fontsize=14)
plt.xlabel('Churn', fontsize=12)
plt.ylabel('Tenure', fontsize=12)
plt.show()

# Monthly Charges vs. Churn
plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=data_cleaned, palette='Set2')
plt.title('Monthly Charges vs. Churn', fontsize=14)
plt.xlabel('Churn', fontsize=12)
plt.ylabel('Monthly Charges', fontsize=12)
plt.show()

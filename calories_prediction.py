# -*- coding: utf-8 -*-
"""
Created on Fri May 16 07:56:46 2025

@author: Bob
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Define file paths
train_path = 'inputs/train.csv'
test_path = 'inputs/test.csv'

# Read the CSV files
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Show basic info
print("Train Data:")
print(train_df.head())

print("\nTest Data:")
print(test_df.head())


# Define expected columns and types
expected_columns = {
    'id': int,
    'Sex': str,
    'Age': float,
    'Height': float,
    'Weight': float,
    'Duration': float,
    'Heart_Rate': float,
    'Body_Temp': float,
    'Calories': float  # Only in train
}

def validate_dataframe(df, name, check_target=True):
    print(f"\nValidating {name} data...")

    # 1. Check for missing values
    missing = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing[missing > 0] if missing.any() else "None")

    # 2. Check for invalid types (not strict but gives a hint)
    print("\nColumn data types:")
    print(df.dtypes)

    # 3. Check for negative or out-of-range values (domain-specific)
    print("\nSanity checks:")
    print("Negative Ages:", (df['Age'] < 0).sum())
    print("Unrealistic Heights (<50cm or >250cm):", ((df['Height'] < 50) | (df['Height'] > 250)).sum())
    print("Unrealistic Weights (<20kg or >300kg):", ((df['Weight'] < 20) | (df['Weight'] > 300)).sum())
    print("Duration <= 0:", (df['Duration'] <= 0).sum())
    print("Heart Rate (<40 or >220):", ((df['Heart_Rate'] < 40) | (df['Heart_Rate'] > 220)).sum())
    print("Body Temp (<35°C or >42°C):", ((df['Body_Temp'] < 35) | (df['Body_Temp'] > 42)).sum())
    
    if check_target:
        print("Calories <= 0:", (df['Calories'] <= 0).sum())

# Run validation
validate_dataframe(train_df, "Train", check_target=True)
validate_dataframe(test_df, "Test", check_target=False)

# Correlation heatmap
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Boxplot of Calories by Sex
sns.boxplot(x='Sex', y='Calories', data=train_df)
plt.title('Calories by Sex')
plt.show()


train_encoded = pd.get_dummies(train_df, columns=['Sex'], drop_first=True)
test_encoded = pd.get_dummies(test_df, columns=['Sex'], drop_first=True)


X = train_encoded.drop(['id', 'Calories'], axis=1)
y = train_encoded['Calories']

# Split into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_val)
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred)))
print("MAE:", mean_absolute_error(y_val, y_pred))
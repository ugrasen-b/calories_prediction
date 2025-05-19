# -*- coding: utf-8 -*-
"""
Created on Mon May 19 18:16:36 2025

@author: ugras
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
# scikit learn regression does not provide the regression table
import statsmodels.api as sm

#%%
# torch imports
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pilpeline

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

# Validate dataframes
def validate_dataframe(df, name, check_target=True):
    print(f"\nValidating {name} data...")

    # 1. Check for missing values
    missing = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing[missing > 0] if missing.any() else "None")

    # 2. Check types
    print("\nColumn data types:")
    print(df.dtypes)

    # 3. Sanity checks
    print("\nSanity checks:")
    print("Negative Ages:", (df['Age'] < 0).sum())
    print("Unrealistic Heights (<50cm or >250cm):", ((df['Height'] < 50) | (df['Height'] > 250)).sum())
    print("Unrealistic Weights (<20kg or >300kg):", ((df['Weight'] < 20) | (df['Weight'] > 300)).sum())
    print("Duration <= 0:", (df['Duration'] <= 0).sum())
    print("Heart Rate (<40 or >220):", ((df['Heart_Rate'] < 40) | (df['Heart_Rate'] > 220)).sum())
    print("Body Temp (<35°C or >42°C):", ((df['Body_Temp'] < 35) | (df['Body_Temp'] > 42)).sum())

    if check_target:
        print("Calories <= 0:", (df['Calories'] <= 0).sum())

validate_dataframe(train_df, "Train", check_target=True)
validate_dataframe(test_df, "Test", check_target=False)

#%%
class CalorieDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
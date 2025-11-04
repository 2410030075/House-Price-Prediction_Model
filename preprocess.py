import pandas as pd
import numpy as np
df=pd.read_csv('Melbourne.csv')
#Display the first and last few rows
print("=== Head (5) ===")
print(df.head())
print("=== Tail (5) ===")
print(df.tail())
print()
#Get information about the dataset
df.info()
#Check for missing values
print("=== Missing values (initial) ===")
print(df.isna().sum())
print()
#Basic statistical information
print("=== Describe (numerical columns) ===")
print(df.describe())
print()
#Include basic stats for object columns as well
if df.select_dtypes(include=["object", "category"]).shape[1] > 0:
    print("=== Describe (categorical columns) ===")
    print(df.describe(include=["object", "category"]))
#Replace None with np.nan
df = df.replace({None: np.nan})
#Identify missing values
print("=== Missing values (before imputation) ===")
print(df.isna().sum())
print()
# Identify column types
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.columns.difference(numeric_cols)

# Impute numeric columns with median
if len(numeric_cols) > 0:
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Impute categorical columns with mode (most frequent)
if len(categorical_cols) > 0:
    modes = df[categorical_cols].mode(dropna=True)
    if not modes.empty:
        df[categorical_cols] = df[categorical_cols].fillna(modes.iloc[0])

# Show missing values after imputation
print("=== Missing values (after imputation) ===")
print(df.isna().sum())
print()

# Calculate Q1, Q3, and IQR for numeric columns only, then cap outliers to median (per column)
if len(numeric_cols) > 0:
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    for col in numeric_cols:
        lower = Q1[col] - 1.5 * IQR[col]
        upper = Q3[col] + 1.5 * IQR[col]
        med = df[col].median()
        mask = (df[col] < lower) | (df[col] > upper)
        if mask.any():
            df.loc[mask, col] = med

# Save the cleaned dataset
output_path = "Melbourne_imputed.csv"
df.to_csv(output_path, index=False)
print(f"Saved cleaned dataset to {output_path}")

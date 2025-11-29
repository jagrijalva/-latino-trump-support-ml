"""
01_data_exploration.py
Load CMPS 2016 data and explore variable names for analysis.
"""

import pandas as pd

# Load the 2016 CMPS data (converted from RDA to CSV via R)
print("Loading CMPS 2016 data...")
df = pd.read_csv('data/CMPS_2016.csv', low_memory=False)

print(f"\nDataset: CMPS 2016")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# List all variable names
print("\n" + "="*60)
print("ALL VARIABLE NAMES")
print("="*60)

for i, col in enumerate(df.columns, 1):
    # Show dtype and sample values for context
    dtype = df[col].dtype
    n_unique = df[col].nunique()
    n_missing = df[col].isna().sum()
    print(f"{i:3}. {col:<40} | {str(dtype):<10} | unique: {n_unique:>5} | missing: {n_missing:>4}")

# Save variable list to file for reference
print("\n" + "="*60)
print("Saving variable list to outputs/variable_list.txt")
print("="*60)

with open('outputs/variable_list.txt', 'w') as f:
    f.write(f"CMPS 2016 Variables\n")
    f.write(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n\n")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        n_missing = df[col].isna().sum()
        f.write(f"{i:3}. {col:<40} | {str(dtype):<10} | unique: {n_unique:>5} | missing: {n_missing:>4}\n")

print("\nDone! Review the variables above to select your DV and predictors.")

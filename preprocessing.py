"""
PHASE 1: Data Preprocessing for Latino Trump Support ML Analysis
================================================================
Step 1.1: Load Data

This script loads the CMPS 2016 raw data and prepares it for analysis.

DV Definition:
- Trump voters = 1
- Non-Trump voters = 0
- Abstainers are excluded from analysis
"""

import pandas as pd
import numpy as np
import subprocess
import os

# =============================================================================
# Step 1.1: Load Data
# =============================================================================

print("=" * 60)
print("PHASE 1: Data Preprocessing")
print("Step 1.1: Load Data")
print("=" * 60)

# Load the RDA file using R to convert to CSV first (handles encoding better)
print("\nLoading CMPS_2016_raw.rda...")

# First, try using pyreadr with error handling
try:
    import pyreadr
    result = pyreadr.read_r('CMPS_2016_raw.rda')
except Exception as e:
    print(f"pyreadr failed with: {e}")
    print("Trying alternative method using R...")

    # Use R to convert RDA to CSV
    r_script = '''
    load("CMPS_2016_raw.rda")
    obj_names <- ls()
    for(name in obj_names) {
        obj <- get(name)
        if(is.data.frame(obj)) {
            write.csv(obj, "temp_cmps_2016.csv", row.names=FALSE, fileEncoding="UTF-8")
            cat(name)
            break
        }
    }
    '''

    with open('temp_convert.R', 'w') as f:
        f.write(r_script)

    result_proc = subprocess.run(['Rscript', 'temp_convert.R'], capture_output=True, text=True)
    if result_proc.returncode != 0:
        print(f"R conversion failed: {result_proc.stderr}")
        raise Exception("Could not load RDA file")

    df_name = result_proc.stdout.strip()
    df = pd.read_csv('temp_cmps_2016.csv', low_memory=False)

    # Clean up temp files
    os.remove('temp_convert.R')
    os.remove('temp_cmps_2016.csv')

    result = {df_name: df}

# Get the dataframe (pyreadr returns a dict with dataframe names as keys)
print(f"\nDataframes found in RDA file: {list(result.keys())}")

# Get the first (and likely only) dataframe
df_name = list(result.keys())[0]
df = result[df_name]

# Document initial dimensions
print("\n" + "-" * 60)
print("INITIAL DATA DIMENSIONS")
print("-" * 60)
print(f"Dataset name: {df_name}")
print(f"Number of rows (observations): {df.shape[0]:,}")
print(f"Number of columns (variables): {df.shape[1]:,}")

# Display basic info
print("\n" + "-" * 60)
print("DATA TYPES SUMMARY")
print("-" * 60)
print(df.dtypes.value_counts())

# Display first few column names
print("\n" + "-" * 60)
print("FIRST 20 COLUMN NAMES")
print("-" * 60)
for i, col in enumerate(df.columns[:20]):
    print(f"  {i+1}. {col}")

print("\n" + "-" * 60)
print("MEMORY USAGE")
print("-" * 60)
print(f"Total memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "=" * 60)
print("Step 1.1 Complete: Data loaded successfully")
print("=" * 60)

# =============================================================================
# Step 1.2: Identify Presidential Vote Variable
# =============================================================================

print("\n" + "=" * 60)
print("Step 1.2: Identify Presidential Vote Variable")
print("=" * 60)

# Search for vote-related columns
vote_keywords = ['vote', 'pres', 'trump', 'clinton', 'elect', 'ballot']
print("\nSearching for vote-related columns...")

vote_cols = []
for col in df.columns:
    col_lower = col.lower()
    if any(kw in col_lower for kw in vote_keywords):
        vote_cols.append(col)

print(f"\nFound {len(vote_cols)} potential vote-related columns:")
for col in vote_cols[:30]:  # Show first 30
    print(f"  - {col}")

# Look at common presidential vote variable names
common_vote_vars = ['PRES_VOTE', 'VOTE_PRES', 'Q20', 'Q21', 'PRES16', 'VOTE2016']
print("\n" + "-" * 60)
print("Checking common presidential vote variable names...")
for var in common_vote_vars:
    if var in df.columns:
        print(f"\nFound: {var}")
        print(df[var].value_counts(dropna=False))

# Check for pattern like Q followed by numbers (survey questions)
print("\n" + "-" * 60)
print("Checking variables with 'PRES' in name:")
pres_vars = [c for c in df.columns if 'PRES' in c.upper()]
for var in pres_vars[:10]:
    print(f"\n{var}:")
    print(df[var].value_counts(dropna=False).head(10))

# List all columns to find vote choice
print("\n" + "-" * 60)
print("ALL COLUMN NAMES (searching for vote-related):")
print("-" * 60)
for i, col in enumerate(df.columns):
    # Only print columns that might be vote-related
    col_str = str(col).upper()
    if any(kw in col_str for kw in ['VOTE', 'PRES', 'TRUMP', 'CLINTON', 'CAND', 'BALLOT', 'ELECT']):
        print(f"  {col}")

# Check columns starting with specific patterns
print("\n" + "-" * 60)
print("Looking for columns with patterns like S1_, Q, or containing candidate names:")
for col in df.columns:
    # Check for Trump or Clinton in the actual values
    if df[col].dtype == 'object':
        values_str = ' '.join(df[col].dropna().astype(str).unique()[:20]).upper()
        if 'TRUMP' in values_str or 'CLINTON' in values_str:
            print(f"\nColumn '{col}' contains Trump/Clinton:")
            print(df[col].value_counts(dropna=False).head(15))

# =============================================================================
# Step 1.2b: Create Dependent Variable (DV)
# =============================================================================

print("\n" + "=" * 60)
print("Step 1.2b: Create Dependent Variable")
print("=" * 60)

# The presidential vote variable is C14
print("\nPresidential Vote Variable (C14) - Full Distribution:")
print(df['C14'].value_counts(dropna=False))

# Create DV: Trump voters (1) vs Non-Trump voters (0)
# Exclude abstainers/non-voters (those who didn't vote are excluded)
print("\n" + "-" * 60)
print("Creating DV: Trump voters (1) vs Non-Trump voters (0)")
print("-" * 60)

# Identify voters only (those who chose a candidate)
# Values containing candidate names indicate actual voters
def create_trump_dv(vote_choice):
    if pd.isna(vote_choice):
        return np.nan  # Will be excluded
    vote_str = str(vote_choice).upper()
    if 'DONALD TRUMP' in vote_str:
        return 1  # Trump voter
    elif any(name in vote_str for name in ['HILLARY CLINTON', 'GARY JOHNSON', 'JILL STEIN', 'SOMEONE ELSE']):
        return 0  # Non-Trump voter (but still a voter)
    else:
        return np.nan  # Unknown/abstainer - exclude

df['trump_vote'] = df['C14'].apply(create_trump_dv)

print("\nDV Distribution (before dropping non-voters):")
print(df['trump_vote'].value_counts(dropna=False))

# Filter to voters only (exclude abstainers/missing)
df_voters = df[df['trump_vote'].notna()].copy()

print(f"\n" + "-" * 60)
print("FINAL DATASET DIMENSIONS (Voters Only)")
print("-" * 60)
print(f"Original observations: {len(df):,}")
print(f"Voters retained: {len(df_voters):,}")
print(f"Non-voters excluded: {len(df) - len(df_voters):,}")

print("\nFinal DV Distribution:")
print(df_voters['trump_vote'].value_counts())

print(f"\nTrump voters: {df_voters['trump_vote'].sum():,.0f} ({df_voters['trump_vote'].mean()*100:.1f}%)")
print(f"Non-Trump voters: {len(df_voters) - df_voters['trump_vote'].sum():,.0f} ({(1-df_voters['trump_vote'].mean())*100:.1f}%)")

# =============================================================================
# Step 1.3: Preserve Survey Weights
# =============================================================================

print("\n" + "=" * 60)
print("Step 1.3: Preserve Survey Weights")
print("=" * 60)

# Extract survey weights as separate vector
survey_wt = df_voters['WEIGHT'].copy()

print(f"\nSurvey weight variable: WEIGHT")
print(f"Weight statistics:")
print(f"  - N: {survey_wt.notna().sum():,}")
print(f"  - Missing: {survey_wt.isna().sum():,}")
print(f"  - Mean: {survey_wt.mean():.4f}")
print(f"  - Std: {survey_wt.std():.4f}")
print(f"  - Min: {survey_wt.min():.4f}")
print(f"  - Max: {survey_wt.max():.4f}")

# Save weights separately for modeling
survey_wt.to_csv('survey_weights.csv', index=False, header=['survey_wt'])
print(f"\nSaved: survey_weights.csv ({len(survey_wt):,} weights)")

# =============================================================================
# Step 1.4: Exclude Inappropriate Variables
# =============================================================================

print("\n" + "=" * 60)
print("Step 1.4: Exclude Inappropriate Variables")
print("=" * 60)

cols_before = len(df_voters.columns)

# Define exclusion categories
exclude_vars = {
    'tautological': ['C6', 'C7', 'C15'],  # Trump/Pence favorability, congressional vote
    'identifiers': ['RESPID', 'ZIPCODE', 'CITY_NAME', 'COUNTY_NAME'],
    'metadata': ['INTERVIEW_START', 'INTERVIEW_END', 'DIFF_DATE', 'ETHNIC_QUOTA'],
    'weights': ['WEIGHT', 'NAT_WEIGHT'],  # Used separately, not as predictors
    'dv_source': ['C14'],  # Already encoded as trump_vote
}

# Identify race-specific variables (A*, B*, BW* items not asked of Latinos)
race_specific_prefixes = ['A', 'B', 'BW']
race_specific_vars = []
for col in df_voters.columns:
    for prefix in race_specific_prefixes:
        # Match columns that start with prefix followed by a number or underscore
        if col.startswith(prefix) and len(col) > len(prefix):
            next_char = col[len(prefix)]
            if next_char.isdigit() or next_char == '_':
                # Check if 100% missing (race-specific by design)
                if df_voters[col].isna().mean() > 0.99:
                    race_specific_vars.append(col)
                    break

exclude_vars['race_specific'] = race_specific_vars

# Identify high missingness variables (>50% missing)
high_missing_vars = []
for col in df_voters.columns:
    if col not in [v for vals in exclude_vars.values() for v in vals]:
        missing_pct = df_voters[col].isna().mean()
        if missing_pct > 0.50:
            high_missing_vars.append(col)

exclude_vars['high_missingness'] = high_missing_vars

# Identify open-text variables (>50 unique values for object type)
open_text_vars = []
for col in df_voters.columns:
    if col not in [v for vals in exclude_vars.values() for v in vals]:
        if df_voters[col].dtype == 'object':
            n_unique = df_voters[col].nunique()
            if n_unique > 50:
                open_text_vars.append(col)

exclude_vars['open_text'] = open_text_vars

# Print exclusion summary
print("\nVariables to exclude by category:")
total_excluded = 0
for category, vars_list in exclude_vars.items():
    # Filter to only variables that exist in the dataframe
    existing_vars = [v for v in vars_list if v in df_voters.columns]
    print(f"\n  {category.upper()} ({len(existing_vars)} variables):")
    if len(existing_vars) <= 10:
        for v in existing_vars:
            print(f"    - {v}")
    else:
        for v in existing_vars[:5]:
            print(f"    - {v}")
        print(f"    ... and {len(existing_vars) - 5} more")
    total_excluded += len(existing_vars)

# Create flat list of all variables to exclude
all_exclude = []
for vars_list in exclude_vars.values():
    all_exclude.extend([v for v in vars_list if v in df_voters.columns])
all_exclude = list(set(all_exclude))  # Remove duplicates

# Remove excluded variables
df_clean = df_voters.drop(columns=all_exclude, errors='ignore')

print(f"\n" + "-" * 60)
print("EXCLUSION SUMMARY")
print("-" * 60)
print(f"Variables before exclusion: {cols_before:,}")
print(f"Variables excluded: {len(all_exclude):,}")
print(f"Variables retained: {len(df_clean.columns):,}")

# Verify trump_vote is still present
if 'trump_vote' in df_clean.columns:
    print(f"\nDV 'trump_vote' retained: YES")
else:
    print(f"\nWARNING: DV 'trump_vote' was excluded!")

# Save the cleaned dataset
print("\n" + "-" * 60)
print("Saving preprocessed data...")
print("-" * 60)
df_clean.to_csv('cmps_2016_clean.csv', index=False)
print(f"Saved: cmps_2016_clean.csv ({len(df_clean):,} rows, {len(df_clean.columns):,} columns)")

# Also save the exclusion log
exclusion_log = pd.DataFrame([
    {'category': cat, 'variable': var}
    for cat, vars_list in exclude_vars.items()
    for var in vars_list if var in df_voters.columns
])
exclusion_log.to_csv('excluded_variables.csv', index=False)
print(f"Saved: excluded_variables.csv ({len(exclusion_log):,} exclusions logged)")

print("\n" + "=" * 60)
print("PHASE 1 COMPLETE: Data Preprocessing")
print("=" * 60)
print(f"\nSummary:")
print(f"  - Loaded CMPS 2016 data: {df.shape[0]:,} observations, {df.shape[1]:,} variables")
print(f"  - Created DV: trump_vote (1=Trump, 0=Non-Trump)")
print(f"  - Filtered to voters only: {len(df_voters):,} observations")
print(f"  - Preserved survey weights: survey_weights.csv")
print(f"  - Excluded {len(all_exclude):,} inappropriate variables")
print(f"  - Final dataset: {len(df_clean):,} obs x {len(df_clean.columns):,} vars")
print(f"  - Saved to: cmps_2016_clean.csv")

#!/usr/bin/env python3
"""
Phase 2 Step 2.7: FINAL Non-Partisan Model Analysis
Latino Trump Support ML Analysis - CMPS 2016

This script:
1. Adds LA203 (mother's partisan affiliation) to exclusion list
2. Re-runs non-partisan model
3. Creates themed Top 30 table
4. Runs SHAP analysis with plots
5. Creates theme summary table
6. Verifies no remaining partisan leaks
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PHASE 2 Step 2.7: FINAL Non-Partisan Model Analysis")
print("=" * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading data...")

X = pd.read_parquet('cmps_2016_X.parquet')
y = pd.read_parquet('cmps_2016_y.parquet')['trump_vote']
weights = pd.read_parquet('cmps_2016_weights.parquet')['survey_wt']

print(f"  X shape: {X.shape[0]:,} rows x {X.shape[1]:,} features")
print(f"  DV distribution: Trump=1: {y.sum():,} ({100*y.mean():.1f}%), Non-Trump=0: {(~y.astype(bool)).sum():,.0f} ({100*(1-y.mean()):.1f}%)")

# =============================================================================
# EXPANDED EXCLUSION LIST (NOW INCLUDING LA203)
# =============================================================================
print("\n" + "=" * 70)
print("EXPANDED EXCLUSION LIST (Including LA203)")
print("=" * 70)

# Prefixes to exclude - NOW INCLUDES LA203
exclude_prefixes = [
    # Candidate Favorability (C2-C11 series)
    'C2_', 'C3_', 'C4_', 'C5_', 'C8_', 'C9_', 'C10_', 'C11_',

    # Party Identification
    'C25_', 'C26_', 'C27_', 'C31_',

    # Party Evaluations
    'L46_', 'L266_', 'L267_',

    # Party Favorability Scales
    'L293_', 'L294_',

    # Derived Party ID
    'C242_HID_',

    # Party Support (LA204 - respondent's group party support)
    'LA204_',

    # NEW: Mother's partisan affiliation (LA203)
    'LA203_',
]

# Exact matches
exclude_exact = ['C242_HID']

def get_exclusion_columns(df_columns, exclude_prefixes, exclude_exact):
    """Identify all columns that should be excluded."""
    exclude_cols = []
    for col in df_columns:
        for prefix in exclude_prefixes:
            if col.startswith(prefix):
                exclude_cols.append(col)
                break
        if col in exclude_exact and col not in exclude_cols:
            exclude_cols.append(col)
    return sorted(list(set(exclude_cols)))

cols_to_exclude = get_exclusion_columns(X.columns, exclude_prefixes, exclude_exact)

print(f"\nTotal columns to exclude: {len(cols_to_exclude)}")

# Count by prefix
prefix_counts = {}
for col in cols_to_exclude:
    for prefix in exclude_prefixes:
        if col.startswith(prefix):
            prefix_key = prefix.rstrip('_')
            prefix_counts[prefix_key] = prefix_counts.get(prefix_key, 0) + 1
            break

print("\nBreakdown by variable type:")
for prefix, count in sorted(prefix_counts.items()):
    print(f"  {prefix}: {count} columns")

# Save excluded columns
pd.DataFrame({'excluded_column': cols_to_exclude}).to_csv('excluded_partisan_columns_final.csv', index=False)

# =============================================================================
# TRAIN/TEST SPLIT
# =============================================================================
print("\n" + "=" * 70)
print("TRAIN/TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, weights, test_size=0.20, stratify=y, random_state=42
)

print(f"  Train: {len(y_train):,} ({100*y_train.mean():.1f}% Trump)")
print(f"  Test:  {len(y_test):,} ({100*y_test.mean():.1f}% Trump)")

# =============================================================================
# NON-PARTISAN MODEL (Updated with LA203 excluded)
# =============================================================================
print("\n" + "=" * 70)
print("NON-PARTISAN MODEL (LA203 Now Excluded)")
print("=" * 70)

# Create non-partisan feature set
X_nonpartisan = X.drop(columns=cols_to_exclude, errors='ignore')
print(f"\n  Original features: {X.shape[1]:,}")
print(f"  After excluding partisan: {X_nonpartisan.shape[1]:,}")
print(f"  Features removed: {X.shape[1] - X_nonpartisan.shape[1]}")

# Split
X_train_np = X_train.drop(columns=cols_to_exclude, errors='ignore')
X_test_np = X_test.drop(columns=cols_to_exclude, errors='ignore')

# Train
rf_params = {
    'n_estimators': 500,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

print("\nTraining non-partisan model...")
rf_nonpartisan = RandomForestClassifier(**rf_params)
rf_nonpartisan.fit(X_train_np, y_train, sample_weight=weights_train)

# Evaluate
y_pred_np = rf_nonpartisan.predict_proba(X_test_np)[:, 1]
auc_nonpartisan = roc_auc_score(y_test, y_pred_np, sample_weight=weights_test)
print(f"  Non-Partisan Model Test ROC-AUC: {auc_nonpartisan:.4f}")

# =============================================================================
# PERMUTATION IMPORTANCE
# =============================================================================
print("\n" + "-" * 60)
print("Permutation Importance - Non-Partisan Model (50 repeats)")
print("-" * 60)
print("  Computing permutation importance...")

perm_imp_np = permutation_importance(
    rf_nonpartisan, X_test_np, y_test,
    n_repeats=50,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)

# Create sorted importance DataFrame
imp_np_df = pd.DataFrame({
    'feature': X_nonpartisan.columns,
    'importance_mean': perm_imp_np.importances_mean,
    'importance_std': perm_imp_np.importances_std
}).sort_values('importance_mean', ascending=False)

top30_np = imp_np_df.head(30).copy()
top30_np['rank'] = range(1, 31)
top30_np = top30_np[['rank', 'feature', 'importance_mean', 'importance_std']]

print("\n  Top 30 Features (Non-Partisan Model):")
print("  " + "-" * 75)
print(f"  {'Rank':<6}{'Feature':<50}{'Importance':<12}{'Std':<10}")
print("  " + "-" * 75)
for _, row in top30_np.iterrows():
    print(f"  {int(row['rank']):<6}{row['feature'][:48]:<50}{row['importance_mean']:.4f}       {row['importance_std']:.4f}")

# =============================================================================
# VERIFICATION: Check for ANY remaining partisan leaks
# =============================================================================
print("\n" + "=" * 70)
print("VERIFICATION: Checking for ANY Partisan Leakage")
print("=" * 70)

# Extended list of partisan prefixes to check
partisan_prefixes_check = [
    'C2_', 'C3_', 'C4_', 'C5_', 'C8_', 'C9_', 'C10_', 'C11_',
    'C25_', 'C26_', 'C27_', 'C31_', 'C242',
    'L46_', 'L266_', 'L267_', 'L293_', 'L294_',
    'LA203_', 'LA204_'
]

leaked_vars = []
for feature in top30_np['feature']:
    for prefix in partisan_prefixes_check:
        if feature.startswith(prefix):
            leaked_vars.append(feature)
            break

if leaked_vars:
    print("\n  WARNING: Partisan variables found in top 30:")
    for var in leaked_vars:
        print(f"    - {var}")
else:
    print("\n  VERIFICATION PASSED: No partisan variables in top 30")

# =============================================================================
# THEME CODING
# =============================================================================
print("\n" + "=" * 70)
print("THEME CODING FOR TOP 30 PREDICTORS")
print("=" * 70)

# Define theme mappings based on variable prefixes
theme_mapping = {
    # Immigration Policy
    'C41_': ('Immigration Policy', 'Govt should reduce immigration'),
    'C158_': ('Immigration Policy', 'Citizenship pathway for undocumented'),
    'C337_': ('Immigration Policy', 'Immigration levels should increase/decrease'),
    'C338_': ('Immigration Policy', 'Number of refugees should...'),
    'C38_': ('Immigration Policy', 'Undocumented should leave jobs'),
    'BLA207_': ('Immigration Policy', 'Federal govt should arrest undocumented'),
    'L241_': ('Immigration Policy', 'Support for deportation policy'),
    'C141_': ('Immigration Policy', 'Support deporting immigrants'),

    # Healthcare Policy
    'C45_': ('Healthcare Policy', 'Support for ACA/Obamacare'),

    # Discrimination Minimization
    'C246_': ('Discrimination Minimization', 'Discrimination against Latinos'),
    'C247_': ('Discrimination Minimization', 'Discrimination against Blacks'),
    'C248_': ('Discrimination Minimization', 'Discrimination against immigrants'),

    # Racism Perception
    'BL155_': ('Racism Perception', 'Racism exists but not major problem'),

    # Cultural Assimilationism / Spanish Language Reaction
    'C142_': ('Cultural Identity', 'Bothered by hearing Spanish spoken'),

    # Emotional Response to Discrimination
    'C111_': ('Emotional Response', 'Feel angry about discrimination'),
    'C112_': ('Emotional Response', 'Feel hopeless about discrimination'),
    'C114_': ('Emotional Response', 'Feel afraid about discrimination'),
    'C115_': ('Emotional Response', 'Feel frustrated about discrimination'),

    # System Trust / Security
    'BL175_': ('System Trust', 'Police treat all races equally'),
    'BL89_': ('System Trust', 'Ever been stopped by police unfairly'),

    # Social Policy / Government Role
    'C40_': ('Social Policy', 'Govt should ensure equal opportunity'),
    'C228_': ('Social Policy', 'Support for gun control'),

    # Economic Policy
    'C256_': ('Economic', 'Own home'),
    'C42_': ('Environmental Policy', 'Climate change legislation'),

    # Census/Demographic
    'SC2011_': ('Demographic', 'Census tract education level'),
    'DP2010_': ('Demographic', 'Census tract age composition'),
}

def get_theme(feature_name):
    """Get theme for a feature based on prefix matching."""
    for prefix, (theme, description) in theme_mapping.items():
        if feature_name.startswith(prefix):
            return theme, description
    return 'Other', 'Unknown/Other'

# Add theme to top 30
top30_themed = top30_np.copy()
top30_themed['theme'] = top30_themed['feature'].apply(lambda x: get_theme(x)[0])
top30_themed['question'] = top30_themed['feature'].apply(lambda x: get_theme(x)[1])

# Extract response from feature name (text after underscore and parenthesis)
def extract_response(feature):
    """Extract response category from one-hot encoded feature name."""
    if '_(' in feature:
        start = feature.find('_(') + 2
        end = feature.rfind(')')
        if end > start:
            return feature[start:end]
    if '_' in feature:
        parts = feature.split('_', 1)
        if len(parts) > 1:
            return parts[1]
    return feature

top30_themed['response'] = top30_themed['feature'].apply(extract_response)

# Display themed table
print("\n  Top 30 Predictors with Theme Coding:")
print("  " + "-" * 90)
print(f"  {'Rank':<5}{'Theme':<25}{'Question':<30}{'Response':<25}{'Imp':<8}")
print("  " + "-" * 90)
for _, row in top30_themed.iterrows():
    print(f"  {int(row['rank']):<5}{row['theme'][:24]:<25}{row['question'][:29]:<30}{row['response'][:24]:<25}{row['importance_mean']:.4f}")

# Save themed results
top30_themed.to_csv('top30_nonpartisan_themed.csv', index=False)
print(f"\n  Saved: top30_nonpartisan_themed.csv")

# =============================================================================
# THEME SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 70)
print("THEME SUMMARY TABLE")
print("=" * 70)

theme_summary = top30_themed.groupby('theme').agg({
    'feature': 'count',
    'importance_mean': 'sum',
    'importance_std': lambda x: np.sqrt(np.sum(x**2))  # Combined std
}).rename(columns={
    'feature': 'count',
    'importance_mean': 'combined_importance',
    'importance_std': 'combined_std'
}).sort_values('combined_importance', ascending=False)

print("\n  Theme-Level Importance Summary:")
print("  " + "-" * 70)
print(f"  {'Theme':<30}{'Count':<8}{'Combined Importance':<20}{'Combined Std':<15}")
print("  " + "-" * 70)
for theme, row in theme_summary.iterrows():
    print(f"  {theme:<30}{int(row['count']):<8}{row['combined_importance']:.4f}{'':<12}{row['combined_std']:.4f}")

# Save theme summary
theme_summary.to_csv('theme_summary.csv')
print(f"\n  Saved: theme_summary.csv")

# =============================================================================
# SHAP ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("SHAP ANALYSIS")
print("=" * 70)

try:
    import shap

    print("\n  Computing SHAP values (this may take a few minutes)...")

    # Use TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(rf_nonpartisan)

    # Compute SHAP values on test set (use subset for speed)
    shap_sample_size = min(500, len(X_test_np))
    X_shap = X_test_np.iloc[:shap_sample_size]
    shap_values = explainer.shap_values(X_shap)

    # For binary classification, shap_values[1] is for positive class (Trump=1)
    if isinstance(shap_values, list):
        shap_values_trump = shap_values[1]
    else:
        shap_values_trump = shap_values

    print(f"  SHAP values computed for {shap_sample_size} test observations")

    # ------------------------------------------
    # SHAP Summary Plot
    # ------------------------------------------
    print("\n  Creating SHAP summary plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_trump, X_shap, max_display=30, show=False)
    plt.title('SHAP Summary: Top 30 Non-Partisan Predictors of Latino Trump Support', fontsize=12)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: shap_summary_plot.png")

    # ------------------------------------------
    # SHAP Bar Plot (mean absolute SHAP)
    # ------------------------------------------
    print("  Creating SHAP bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_trump, X_shap, plot_type="bar", max_display=30, show=False)
    plt.title('Mean |SHAP| Values: Feature Importance', fontsize=12)
    plt.tight_layout()
    plt.savefig('shap_bar_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: shap_bar_plot.png")

    # ------------------------------------------
    # SHAP Dependence Plot for C41 (top predictor)
    # ------------------------------------------
    print("  Creating SHAP dependence plot for C41...")
    c41_cols = [c for c in X_shap.columns if c.startswith('C41_')]
    if c41_cols:
        # Find most important C41 variant
        c41_importance = {c: np.abs(shap_values_trump[:, list(X_shap.columns).index(c)]).mean()
                         for c in c41_cols}
        top_c41 = max(c41_importance, key=c41_importance.get)

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(top_c41, shap_values_trump, X_shap, show=False)
        plt.title(f'SHAP Dependence: {top_c41[:50]}', fontsize=11)
        plt.tight_layout()
        plt.savefig('shap_dependence_c41.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: shap_dependence_c41.png (for {top_c41[:40]}...)")

    # ------------------------------------------
    # SHAP Dependence Plot for C142 (Spanish language)
    # ------------------------------------------
    print("  Creating SHAP dependence plot for C142...")
    c142_cols = [c for c in X_shap.columns if c.startswith('C142_')]
    if c142_cols:
        c142_importance = {c: np.abs(shap_values_trump[:, list(X_shap.columns).index(c)]).mean()
                          for c in c142_cols}
        top_c142 = max(c142_importance, key=c142_importance.get)

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(top_c142, shap_values_trump, X_shap, show=False)
        plt.title(f'SHAP Dependence: {top_c142[:50]}', fontsize=11)
        plt.tight_layout()
        plt.savefig('shap_dependence_c142.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: shap_dependence_c142.png (for {top_c142[:40]}...)")

    # ------------------------------------------
    # SHAP Waterfall Plots for Trump Voters
    # ------------------------------------------
    print("  Creating SHAP waterfall plots for sample Trump voters...")

    # Find Trump voters in test set
    trump_voter_indices = np.where(y_test.iloc[:shap_sample_size].values == 1)[0]

    if len(trump_voter_indices) >= 2:
        for i, idx in enumerate(trump_voter_indices[:2]):
            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(shap.Explanation(
                values=shap_values_trump[idx],
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                data=X_shap.iloc[idx],
                feature_names=X_shap.columns.tolist()
            ), max_display=15, show=False)
            plt.title(f'SHAP Waterfall: Latino Trump Voter #{i+1}', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'shap_waterfall_trump_voter_{i+1}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: shap_waterfall_trump_voter_{i+1}.png")

    # ------------------------------------------
    # Save SHAP importance ranking
    # ------------------------------------------
    shap_importance = pd.DataFrame({
        'feature': X_shap.columns,
        'mean_abs_shap': np.abs(shap_values_trump).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    shap_importance.to_csv('shap_importance_nonpartisan.csv', index=False)
    print("  Saved: shap_importance_nonpartisan.csv")

    print("\n  SHAP analysis complete!")

except ImportError:
    print("\n  WARNING: SHAP not installed. Skipping SHAP analysis.")
    print("  Install with: pip install shap")
except Exception as e:
    print(f"\n  WARNING: SHAP analysis failed: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"""
NON-PARTISAN MODEL RESULTS (LA203 NOW EXCLUDED)
------------------------------------------------
Features:               {X_nonpartisan.shape[1]:,} (excluded {len(cols_to_exclude)})
Test ROC-AUC:           {auc_nonpartisan:.4f}

VARIABLES EXCLUDED (Total: {len(cols_to_exclude)})
--------------------------------------------------
- Candidate favorability: C2-C11 (all categories)
- Party identification: C25, C26, C27
- Ideology: C31 (liberal-conservative scale)
- Party evaluations: L46, L266, L267
- Party favorability: L293, L294 (0-10 scales)
- Derived party ID: C242_HID
- Party support: LA204 (group party support)
- Mother's partisan affiliation: LA203 (NEW)

VERIFICATION: {'PASSED - No partisan leaks' if not leaked_vars else 'FAILED - See warnings above'}

TOP 5 NON-PARTISAN PREDICTORS
-----------------------------
""")

for _, row in top30_themed.head(5).iterrows():
    print(f"  {int(row['rank'])}. {row['feature'][:50]}")
    print(f"     Theme: {row['theme']} | Importance: {row['importance_mean']:.4f}")

print(f"""
OUTPUT FILES
------------
- top30_nonpartisan_themed.csv (Top 30 with theme coding)
- theme_summary.csv (Theme-level summary)
- excluded_partisan_columns_final.csv (Excluded columns)
- shap_summary_plot.png (SHAP summary)
- shap_bar_plot.png (SHAP bar chart)
- shap_dependence_c41.png (C41 dependence)
- shap_dependence_c142.png (C142 dependence)
- shap_waterfall_trump_voter_1.png (Individual prediction)
- shap_waterfall_trump_voter_2.png (Individual prediction)
- shap_importance_nonpartisan.csv (SHAP rankings)
""")

print("=" * 70)
print("Step 2.7 FINAL Analysis Complete")
print("=" * 70)

#!/usr/bin/env python3
"""
Methods Gaps Analysis
=====================
1. Tautological audit: Find Trump-related variables beyond C4_
2. Evaluation metrics: Confusion matrix, precision/recall/F1, Brier score, CV AUC
3. Imputation cross-reference: Which top 30 predictors had imputed values?
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, classification_report
)
import os

# Create output directory
os.makedirs('outputs/results', exist_ok=True)

print("=" * 70)
print("METHODS GAPS ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. TAUTOLOGICAL AUDIT
# =============================================================================
print("\n" + "=" * 70)
print("1. TAUTOLOGICAL AUDIT: Trump-Related Variables")
print("=" * 70)

# Load feature names
feature_names = pd.read_csv('cmps_2016_feature_names.csv')['feature'].tolist()

# Load excluded partisan columns
excluded_partisan = pd.read_csv('excluded_partisan_columns_final.csv')['excluded_column'].tolist()

# Search for Trump-related patterns in feature names (case insensitive)
trump_patterns = ['trump', 'donald', 'c4_', 'c4(']  # C4 is Trump favorability

kept_trump_vars = []
excluded_trump_vars = []

for feat in feature_names:
    feat_lower = feat.lower()
    if any(pattern.lower() in feat_lower for pattern in trump_patterns):
        if feat in excluded_partisan:
            excluded_trump_vars.append(feat)
        else:
            kept_trump_vars.append(feat)

print("\n--- Trump-Related Variables in Feature Names ---")
print(f"\nVariables containing 'Trump', 'Donald', or 'C4_' prefix:")
print(f"  - EXCLUDED (as partisan): {len(excluded_trump_vars)}")
for v in excluded_trump_vars:
    print(f"    • {v}")

print(f"\n  - KEPT in full model: {len(kept_trump_vars)}")
if kept_trump_vars:
    for v in kept_trump_vars:
        print(f"    • {v}")
else:
    print("    (None found)")

# Also check for C14 (vote choice source) - should be excluded
c14_vars = [f for f in feature_names if f.lower().startswith('c14')]
print(f"\n--- C14 (Vote Choice Source) Variables ---")
print(f"C14 variables in features: {c14_vars if c14_vars else 'None (correctly excluded)'}")

# Save audit results
audit_results = {
    'variable': excluded_trump_vars + kept_trump_vars,
    'type': ['Trump favorability (C4_)'] * len(excluded_trump_vars) +
            ['Other Trump-related'] * len(kept_trump_vars),
    'status': ['EXCLUDED (partisan)'] * len(excluded_trump_vars) +
              ['KEPT (in full model)'] * len(kept_trump_vars)
}
audit_df = pd.DataFrame(audit_results)
audit_df.to_csv('outputs/results/tautological_audit.csv', index=False)
print(f"\nAudit saved to: outputs/results/tautological_audit.csv")

# =============================================================================
# 2. EVALUATION METRICS
# =============================================================================
print("\n" + "=" * 70)
print("2. EVALUATION METRICS")
print("=" * 70)

# Load data
X = pd.read_parquet('cmps_2016_X.parquet')
y = pd.read_parquet('cmps_2016_y.parquet').iloc[:, 0]

# Train/test split (same as original)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define models
rf_params = {
    'n_estimators': 500,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

# Excluded partisan columns for non-partisan model
excluded_partisan = pd.read_csv('excluded_partisan_columns_final.csv')['excluded_column'].tolist()

# Function to compute all metrics
def compute_metrics(y_true, y_pred, y_prob, model_name):
    """Compute all evaluation metrics"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'Model': model_name,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'True Positives': tp,
        'Accuracy': (tp + tn) / (tp + tn + fp + fn),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'Brier Score': brier_score_loss(y_true, y_prob)
    }
    return metrics, cm

# Baseline accuracy (predict all as majority class)
baseline_accuracy = (y_test == 0).mean()
print(f"\n--- Baseline Accuracy ---")
print(f"If we predicted 'Non-Trump' for everyone: {baseline_accuracy:.4f} ({baseline_accuracy*100:.1f}%)")
print(f"Trump voters in test set: {(y_test == 1).sum()} / {len(y_test)} = {(y_test == 1).mean()*100:.1f}%")

# ---- FULL MODEL ----
print("\n--- Full Model ---")
print("Training Full Model...")
rf_full = RandomForestClassifier(**rf_params)
rf_full.fit(X_train, y_train)

y_pred_full = rf_full.predict(X_test)
y_prob_full = rf_full.predict_proba(X_test)[:, 1]

metrics_full, cm_full = compute_metrics(y_test, y_pred_full, y_prob_full, 'Full Model')

print(f"\nConfusion Matrix (threshold=0.5):")
print(f"                  Predicted")
print(f"                  Non-Trump  Trump")
print(f"Actual Non-Trump    {cm_full[0,0]:>5}    {cm_full[0,1]:>5}")
print(f"Actual Trump        {cm_full[1,0]:>5}    {cm_full[1,1]:>5}")
print(f"\nPrecision: {metrics_full['Precision']:.4f}")
print(f"Recall:    {metrics_full['Recall']:.4f}")
print(f"F1 Score:  {metrics_full['F1']:.4f}")
print(f"ROC-AUC:   {metrics_full['ROC-AUC']:.4f}")
print(f"Brier Score: {metrics_full['Brier Score']:.4f}")

# Cross-validation for full model
print("\nRunning 5-fold CV for Full Model...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_full = cross_val_score(rf_full, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"CV AUC: {cv_scores_full.mean():.4f} ± {cv_scores_full.std():.4f}")
print(f"Fold scores: {[f'{s:.4f}' for s in cv_scores_full]}")
metrics_full['CV_AUC_Mean'] = cv_scores_full.mean()
metrics_full['CV_AUC_Std'] = cv_scores_full.std()

# ---- NON-PARTISAN MODEL ----
print("\n--- Non-Partisan Model ---")

# Remove partisan columns
cols_to_keep = [c for c in X.columns if c not in excluded_partisan]
X_nonpartisan = X[cols_to_keep]
X_train_np = X_train[cols_to_keep]
X_test_np = X_test[cols_to_keep]

print(f"Features removed: {len(excluded_partisan)}")
print(f"Features remaining: {len(cols_to_keep)}")
print("Training Non-Partisan Model...")

rf_nonpartisan = RandomForestClassifier(**rf_params)
rf_nonpartisan.fit(X_train_np, y_train)

y_pred_np = rf_nonpartisan.predict(X_test_np)
y_prob_np = rf_nonpartisan.predict_proba(X_test_np)[:, 1]

metrics_np, cm_np = compute_metrics(y_test, y_pred_np, y_prob_np, 'Non-Partisan Model')

print(f"\nConfusion Matrix (threshold=0.5):")
print(f"                  Predicted")
print(f"                  Non-Trump  Trump")
print(f"Actual Non-Trump    {cm_np[0,0]:>5}    {cm_np[0,1]:>5}")
print(f"Actual Trump        {cm_np[1,0]:>5}    {cm_np[1,1]:>5}")
print(f"\nPrecision: {metrics_np['Precision']:.4f}")
print(f"Recall:    {metrics_np['Recall']:.4f}")
print(f"F1 Score:  {metrics_np['F1']:.4f}")
print(f"ROC-AUC:   {metrics_np['ROC-AUC']:.4f}")
print(f"Brier Score: {metrics_np['Brier Score']:.4f}")

# Cross-validation for non-partisan model
print("\nRunning 5-fold CV for Non-Partisan Model...")
cv_scores_np = cross_val_score(rf_nonpartisan, X_nonpartisan, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"CV AUC: {cv_scores_np.mean():.4f} ± {cv_scores_np.std():.4f}")
print(f"Fold scores: {[f'{s:.4f}' for s in cv_scores_np]}")
metrics_np['CV_AUC_Mean'] = cv_scores_np.mean()
metrics_np['CV_AUC_Std'] = cv_scores_np.std()

# Save metrics comparison
metrics_comparison = pd.DataFrame([metrics_full, metrics_np])
metrics_comparison['Baseline_Accuracy'] = baseline_accuracy
metrics_comparison.to_csv('outputs/results/evaluation_metrics.csv', index=False)
print(f"\nMetrics saved to: outputs/results/evaluation_metrics.csv")

# Save confusion matrices
cm_df = pd.DataFrame({
    'Model': ['Full Model', 'Full Model', 'Non-Partisan', 'Non-Partisan'],
    'Actual': ['Non-Trump', 'Trump', 'Non-Trump', 'Trump'],
    'Predicted_NonTrump': [cm_full[0,0], cm_full[1,0], cm_np[0,0], cm_np[1,0]],
    'Predicted_Trump': [cm_full[0,1], cm_full[1,1], cm_np[0,1], cm_np[1,1]]
})
cm_df.to_csv('outputs/results/confusion_matrices.csv', index=False)
print(f"Confusion matrices saved to: outputs/results/confusion_matrices.csv")

# =============================================================================
# 3. IMPUTATION CROSS-REFERENCE
# =============================================================================
print("\n" + "=" * 70)
print("3. IMPUTATION CROSS-REFERENCE")
print("=" * 70)

# Load imputation log
imputation_log = pd.read_csv('cmps_2016_imputation_log.csv')
imputed_vars = set(imputation_log['variable'].tolist())
n_total = len(y)  # Total observations

# Load top 30 from each model
top30_full = pd.read_csv('top30_full_model_corrected.csv')
top30_nonpartisan = pd.read_csv('top30_nonpartisan_themed.csv')

def check_imputation(features_df, model_name):
    """Check which top 30 features had imputation"""
    results = []

    for _, row in features_df.iterrows():
        feature = row['feature']
        # Extract base variable name (before the level indicator)
        # E.g., "C4_(4) Very unfavorable" -> base is "C4"
        if '_(' in feature:
            base_var = feature.split('_(')[0]
        elif '_' in feature and not feature.startswith('DP') and not feature.startswith('HISP') and not feature.startswith('SC') and not feature.startswith('EC') and not feature.startswith('HC'):
            base_var = feature.split('_')[0]
        else:
            base_var = feature

        # Check if this base variable was imputed
        imputed_rows = imputation_log[imputation_log['variable'] == base_var]
        if len(imputed_rows) > 0:
            n_imputed = imputed_rows['n_imputed'].iloc[0]
            pct_imputed = (n_imputed / n_total) * 100
            results.append({
                'rank': row['rank'],
                'feature': feature,
                'base_variable': base_var,
                'was_imputed': True,
                'n_imputed': n_imputed,
                'pct_imputed': pct_imputed
            })
        else:
            # Check for exact match (for census variables)
            imputed_rows = imputation_log[imputation_log['variable'] == feature]
            if len(imputed_rows) > 0:
                n_imputed = imputed_rows['n_imputed'].iloc[0]
                pct_imputed = (n_imputed / n_total) * 100
                results.append({
                    'rank': row['rank'],
                    'feature': feature,
                    'base_variable': feature,
                    'was_imputed': True,
                    'n_imputed': n_imputed,
                    'pct_imputed': pct_imputed
                })
            else:
                results.append({
                    'rank': row['rank'],
                    'feature': feature,
                    'base_variable': base_var,
                    'was_imputed': False,
                    'n_imputed': 0,
                    'pct_imputed': 0.0
                })

    return pd.DataFrame(results)

print("\n--- Full Model Top 30: Imputation Status ---")
imputation_full = check_imputation(top30_full, 'Full Model')
imputed_count_full = imputation_full['was_imputed'].sum()
print(f"Variables with any imputation: {imputed_count_full} / 30")
print("\nImputed variables:")
for _, row in imputation_full[imputation_full['was_imputed']].iterrows():
    print(f"  Rank {row['rank']:>2}: {row['feature'][:50]:<50} - {row['n_imputed']} ({row['pct_imputed']:.1f}%)")

print("\n--- Non-Partisan Model Top 30: Imputation Status ---")
imputation_np = check_imputation(top30_nonpartisan, 'Non-Partisan Model')
imputed_count_np = imputation_np['was_imputed'].sum()
print(f"Variables with any imputation: {imputed_count_np} / 30")
print("\nImputed variables:")
for _, row in imputation_np[imputation_np['was_imputed']].iterrows():
    print(f"  Rank {row['rank']:>2}: {row['feature'][:50]:<50} - {row['n_imputed']} ({row['pct_imputed']:.1f}%)")

# Save imputation cross-reference
imputation_full['model'] = 'Full Model'
imputation_np['model'] = 'Non-Partisan Model'
imputation_combined = pd.concat([imputation_full, imputation_np], ignore_index=True)
imputation_combined.to_csv('outputs/results/top30_imputation_crossref.csv', index=False)
print(f"\nImputation cross-reference saved to: outputs/results/top30_imputation_crossref.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
1. TAUTOLOGICAL AUDIT:
   - Trump-related variables excluded (C4_ favorability): {len(excluded_trump_vars)}
   - Trump-related variables kept beyond C4_: {len(kept_trump_vars)}
   - Status: {'CLEAN - No unexpected Trump variables' if len(kept_trump_vars) == 0 else 'REVIEW NEEDED'}

2. EVALUATION METRICS:
   - Baseline accuracy (predict all Non-Trump): {baseline_accuracy*100:.1f}%

   Full Model:
     • Test AUC: {metrics_full['ROC-AUC']:.4f}
     • CV AUC:   {metrics_full['CV_AUC_Mean']:.4f} ± {metrics_full['CV_AUC_Std']:.4f}
     • Precision/Recall/F1: {metrics_full['Precision']:.3f} / {metrics_full['Recall']:.3f} / {metrics_full['F1']:.3f}
     • Brier Score: {metrics_full['Brier Score']:.4f}

   Non-Partisan Model:
     • Test AUC: {metrics_np['ROC-AUC']:.4f}
     • CV AUC:   {metrics_np['CV_AUC_Mean']:.4f} ± {metrics_np['CV_AUC_Std']:.4f}
     • Precision/Recall/F1: {metrics_np['Precision']:.3f} / {metrics_np['Recall']:.3f} / {metrics_np['F1']:.3f}
     • Brier Score: {metrics_np['Brier Score']:.4f}

3. IMPUTATION CROSS-REFERENCE:
   - Full Model Top 30: {imputed_count_full}/30 had imputed values
   - Non-Partisan Top 30: {imputed_count_np}/30 had imputed values
""")

print("\nOutput files saved to outputs/results/:")
print("  - tautological_audit.csv")
print("  - evaluation_metrics.csv")
print("  - confusion_matrices.csv")
print("  - top30_imputation_crossref.csv")
print("\n" + "=" * 70)

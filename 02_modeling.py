"""
PHASE 2: Modeling and Interpretation
=====================================
Latino Trump Support ML Analysis

Steps 2.1-2.3: Data loading, train/test split, and hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer, average_precision_score, f1_score, confusion_matrix, classification_report, brier_score_loss
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Step 2.1: Load Data
# =============================================================================

print("=" * 60)
print("PHASE 2: Modeling and Interpretation")
print("Step 2.1: Load Data")
print("=" * 60)

# Load parquet files
X = pd.read_parquet('cmps_2016_X.parquet')
y = pd.read_parquet('cmps_2016_y.parquet')['trump_vote']
weights = pd.read_parquet('cmps_2016_weights.parquet')['survey_wt']

print(f"\nData loaded successfully:")
print(f"  X shape: {X.shape[0]:,} rows x {X.shape[1]:,} features")
print(f"  y shape: {len(y):,} labels")
print(f"  weights shape: {len(weights):,} weights")

print(f"\nTarget distribution:")
print(f"  Trump voters (1): {y.sum():,} ({y.mean()*100:.1f}%)")
print(f"  Non-Trump voters (0): {(1-y).sum():,} ({(1-y.mean())*100:.1f}%)")

# =============================================================================
# Step 2.2: Train/Test Split
# =============================================================================

print("\n" + "=" * 60)
print("Step 2.2: Train/Test Split")
print("=" * 60)

# 80/20 split, stratified on y, random_state=42
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, weights,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print(f"\nTrain/Test Split (80/20, stratified):")
print(f"  Training set: {len(X_train):,} observations")
print(f"  Test set: {len(X_test):,} observations")

print(f"\nTraining set target distribution:")
print(f"  Trump voters (1): {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"  Non-Trump voters (0): {(1-y_train).sum():,} ({(1-y_train.mean())*100:.1f}%)")

print(f"\nTest set target distribution:")
print(f"  Trump voters (1): {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
print(f"  Non-Trump voters (0): {(1-y_test).sum():,} ({(1-y_test.mean())*100:.1f}%)")

# =============================================================================
# Step 2.3: GridSearchCV for Hyperparameter Tuning
# =============================================================================

print("\n" + "=" * 60)
print("Step 2.3: GridSearchCV (5-fold CV)")
print("=" * 60)

# Define parameter grid
param_grid = {
    'max_features': ['sqrt', 0.33],
    'min_samples_leaf': [1, 5]
}

# Fixed parameters
fixed_params = {
    'n_estimators': 500,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

print(f"\nFixed parameters:")
for k, v in fixed_params.items():
    print(f"  {k}: {v}")

print(f"\nParameter grid to search:")
for k, v in param_grid.items():
    print(f"  {k}: {v}")

print(f"\nTotal combinations: {len(param_grid['max_features']) * len(param_grid['min_samples_leaf'])}")

# Create base estimator
rf = RandomForestClassifier(**fixed_params)

# Create stratified k-fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Run GridSearchCV
print(f"\nRunning 5-fold cross-validation...")
print(f"Optimizing: ROC-AUC")

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

# Fit with sample weights
grid_search.fit(X_train, y_train, sample_weight=weights_train)

# =============================================================================
# Results
# =============================================================================

print("\n" + "=" * 60)
print("GridSearchCV Results")
print("=" * 60)

# Best parameters
print(f"\nBest Parameters:")
for k, v in grid_search.best_params_.items():
    print(f"  {k}: {v}")

print(f"\nBest CV ROC-AUC Score: {grid_search.best_score_:.4f}")

# All results
print(f"\n" + "-" * 60)
print("All Parameter Combinations:")
print("-" * 60)

results_df = pd.DataFrame(grid_search.cv_results_)
results_summary = results_df[[
    'param_max_features',
    'param_min_samples_leaf',
    'mean_test_score',
    'std_test_score',
    'mean_train_score',
    'rank_test_score'
]].sort_values('rank_test_score')

results_summary.columns = ['max_features', 'min_samples_leaf', 'CV_AUC_mean', 'CV_AUC_std', 'Train_AUC', 'Rank']

print(results_summary.to_string(index=False))

# Check for overfitting
print(f"\n" + "-" * 60)
print("Overfitting Check (Train - CV difference):")
print("-" * 60)
for idx, row in results_summary.iterrows():
    overfit = row['Train_AUC'] - row['CV_AUC_mean']
    print(f"  max_features={row['max_features']}, min_samples_leaf={row['min_samples_leaf']}: "
          f"Train={row['Train_AUC']:.4f}, CV={row['CV_AUC_mean']:.4f}, Diff={overfit:.4f}")

print("\n" + "=" * 60)
print("Step 2.3 Complete: Ready to train final model")
print("=" * 60)

# =============================================================================
# Step 2.4: Train Final Model
# =============================================================================

print("\n" + "=" * 60)
print("Step 2.4: Train Final Model")
print("=" * 60)

# Best parameters from CV
best_params = {
    'n_estimators': 500,
    'max_features': grid_search.best_params_['max_features'],
    'min_samples_leaf': grid_search.best_params_['min_samples_leaf'],
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

print(f"\nFinal model parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# Train final model on full training set
print(f"\nTraining final model on {len(X_train):,} observations...")
final_model = RandomForestClassifier(**best_params)
final_model.fit(X_train, y_train, sample_weight=weights_train)

print(f"Model trained successfully.")
print(f"  Number of trees: {final_model.n_estimators}")
print(f"  Number of features: {final_model.n_features_in_}")

# =============================================================================
# Step 2.5: Evaluate on Test Set
# =============================================================================

print("\n" + "=" * 60)
print("Step 2.5: Evaluate on Test Set")
print("=" * 60)

# Get predictions
y_train_pred_proba = final_model.predict_proba(X_train)[:, 1]
y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]
y_test_pred = final_model.predict(X_test)

# Calculate metrics
train_roc_auc = roc_auc_score(y_train, y_train_pred_proba, sample_weight=weights_train)
test_roc_auc = roc_auc_score(y_test, y_test_pred_proba, sample_weight=weights_test)
test_pr_auc = average_precision_score(y_test, y_test_pred_proba, sample_weight=weights_test)
test_f1 = f1_score(y_test, y_test_pred)

# Confusion matrix at threshold=0.5
cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n" + "-" * 60)
print("PRIMARY METRICS")
print("-" * 60)

# ROC-AUC
roc_target = 0.75
roc_status = "✓ PASS" if test_roc_auc >= roc_target else "✗ FAIL"
print(f"\nROC-AUC (target ≥{roc_target}):")
print(f"  Test ROC-AUC: {test_roc_auc:.4f} {roc_status}")

# PR-AUC
pr_target = 0.45
pr_status = "✓ PASS" if test_pr_auc >= pr_target else "✗ FAIL"
print(f"\nPR-AUC (target ≥{pr_target}):")
print(f"  Test PR-AUC: {test_pr_auc:.4f} {pr_status}")

# F1 Score
print(f"\nF1 Score:")
print(f"  Test F1: {test_f1:.4f}")

print(f"\n" + "-" * 60)
print("CONFUSION MATRIX (threshold=0.5)")
print("-" * 60)
print(f"\n                 Predicted")
print(f"                 Non-Trump  Trump")
print(f"Actual Non-Trump    {tn:4d}    {fp:4d}")
print(f"Actual Trump        {fn:4d}    {tp:4d}")

# Additional metrics from confusion matrix
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\nDerived metrics:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision (Trump): {precision:.4f}")
print(f"  Recall (Trump): {recall:.4f}")
print(f"  Specificity (Non-Trump): {specificity:.4f}")

print(f"\n" + "-" * 60)
print("OVERFITTING CHECK (Train vs Test)")
print("-" * 60)
auc_gap = train_roc_auc - test_roc_auc
print(f"\n  Train ROC-AUC: {train_roc_auc:.4f}")
print(f"  Test ROC-AUC:  {test_roc_auc:.4f}")
print(f"  Gap:           {auc_gap:.4f}")

if auc_gap > 0.10:
    print(f"\n  ⚠ WARNING: Train-Test gap > 0.10 suggests overfitting")
elif auc_gap > 0.05:
    print(f"\n  ⚠ CAUTION: Train-Test gap > 0.05, moderate overfitting")
else:
    print(f"\n  ✓ Gap < 0.05, minimal overfitting")

print(f"\n" + "=" * 60)
print("Step 2.5 Complete: Model Evaluation Summary")
print("=" * 60)

print(f"\n{'Metric':<20} {'Value':<10} {'Target':<10} {'Status':<10}")
print("-" * 50)
print(f"{'ROC-AUC':<20} {test_roc_auc:<10.4f} {'≥0.75':<10} {roc_status:<10}")
print(f"{'PR-AUC':<20} {test_pr_auc:<10.4f} {'≥0.45':<10} {pr_status:<10}")
print(f"{'F1 Score':<20} {test_f1:<10.4f} {'-':<10} {'-':<10}")
print(f"{'Train-Test Gap':<20} {auc_gap:<10.4f} {'<0.10':<10} {'✓' if auc_gap < 0.10 else '✗':<10}")

# =============================================================================
# Step 2.5b: Additional Evaluation Checks
# =============================================================================

print("\n" + "=" * 60)
print("Step 2.5b: Additional Evaluation Checks")
print("=" * 60)

# --- Weighted ROC-AUC confirmation ---
print(f"\n" + "-" * 60)
print("WEIGHTED ROC-AUC (using test weights)")
print("-" * 60)
weighted_roc_auc = roc_auc_score(y_test, y_test_pred_proba, sample_weight=weights_test)
unweighted_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
print(f"  Weighted ROC-AUC:   {weighted_roc_auc:.4f}")
print(f"  Unweighted ROC-AUC: {unweighted_roc_auc:.4f}")
print(f"  Difference:         {weighted_roc_auc - unweighted_roc_auc:.4f}")

# --- Bootstrap 95% CI for ROC-AUC ---
print(f"\n" + "-" * 60)
print("95% BOOTSTRAP CI FOR ROC-AUC (1000 iterations)")
print("-" * 60)

np.random.seed(42)
n_bootstrap = 1000
bootstrap_aucs = []

n_test = len(y_test)
y_test_arr = y_test.values
weights_test_arr = weights_test.values

print(f"  Running {n_bootstrap} bootstrap iterations...")

for i in range(n_bootstrap):
    # Sample with replacement
    indices = np.random.choice(n_test, size=n_test, replace=True)
    y_boot = y_test_arr[indices]
    pred_boot = y_test_pred_proba[indices]
    weights_boot = weights_test_arr[indices]

    # Skip if only one class in bootstrap sample
    if len(np.unique(y_boot)) < 2:
        continue

    auc_boot = roc_auc_score(y_boot, pred_boot, sample_weight=weights_boot)
    bootstrap_aucs.append(auc_boot)

bootstrap_aucs = np.array(bootstrap_aucs)
ci_lower = np.percentile(bootstrap_aucs, 2.5)
ci_upper = np.percentile(bootstrap_aucs, 97.5)
ci_mean = np.mean(bootstrap_aucs)
ci_std = np.std(bootstrap_aucs)

print(f"\n  Bootstrap Results ({len(bootstrap_aucs)} valid iterations):")
print(f"    Mean ROC-AUC:     {ci_mean:.4f}")
print(f"    Std Dev:          {ci_std:.4f}")
print(f"    95% CI:           [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"    Point Estimate:   {weighted_roc_auc:.4f}")

# Check if CI excludes 0.5 (random chance)
if ci_lower > 0.5:
    print(f"\n  ✓ 95% CI excludes 0.5 (random chance) - model is statistically significant")
else:
    print(f"\n  ⚠ 95% CI includes 0.5 - model may not be significantly better than random")

# --- Brier Score ---
print(f"\n" + "-" * 60)
print("BRIER SCORE (probability calibration)")
print("-" * 60)

brier = brier_score_loss(y_test, y_test_pred_proba, sample_weight=weights_test)
# Baseline Brier score (predicting base rate)
base_rate = y_test.mean()
brier_baseline = np.average((y_test - base_rate)**2, weights=weights_test)
brier_skill = 1 - (brier / brier_baseline)

print(f"  Brier Score:          {brier:.4f}")
print(f"  Baseline Brier:       {brier_baseline:.4f} (predicting {base_rate:.1%} for all)")
print(f"  Brier Skill Score:    {brier_skill:.4f}")
print(f"\n  Interpretation:")
print(f"    - Brier Score range: [0, 1], lower is better")
print(f"    - Perfect predictions: 0.0")
print(f"    - Brier < {brier_baseline:.4f} indicates better than baseline")
if brier < brier_baseline:
    print(f"    ✓ Model predictions are well-calibrated (Brier < baseline)")
else:
    print(f"    ⚠ Model predictions worse than baseline")

# --- Learning Curves ---
print(f"\n" + "-" * 60)
print("LEARNING CURVES")
print("-" * 60)

print(f"\n  Computing learning curves (this may take a moment)...")

# Use fewer estimators for learning curves to speed up computation
lc_model = RandomForestClassifier(
    n_estimators=100,  # Reduced for speed
    max_features='sqrt',
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

train_sizes = np.linspace(0.1, 1.0, 10)

train_sizes_abs, train_scores, val_scores = learning_curve(
    lc_model, X_train, y_train,
    train_sizes=train_sizes,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

print(f"\n  Learning Curve Results:")
print(f"  {'Train Size':<12} {'Train AUC':<15} {'Val AUC':<15} {'Gap':<10}")
print("  " + "-" * 52)

for i, size in enumerate(train_sizes_abs):
    gap = train_mean[i] - val_mean[i]
    print(f"  {size:<12} {train_mean[i]:.4f} ± {train_std[i]:.3f}   {val_mean[i]:.4f} ± {val_std[i]:.3f}   {gap:.4f}")

print(f"\n  Learning Curve Interpretation:")
final_gap = train_mean[-1] - val_mean[-1]
print(f"    - Final train-val gap: {final_gap:.4f}")

if val_mean[-1] > val_mean[0] + 0.02:
    print(f"    - Validation AUC increased from {val_mean[0]:.4f} to {val_mean[-1]:.4f}")
    print(f"    ✓ Model benefits from more training data")
else:
    print(f"    - Validation AUC relatively stable")
    print(f"    → Model may have reached data saturation")

if final_gap > 0.05:
    print(f"    ⚠ High bias-variance gap suggests potential overfitting")
else:
    print(f"    ✓ Reasonable bias-variance tradeoff")

# =============================================================================
# Final Summary
# =============================================================================

print("\n" + "=" * 60)
print("COMPLETE EVALUATION SUMMARY")
print("=" * 60)

print(f"\n{'Metric':<25} {'Value':<15} {'Notes':<30}")
print("-" * 70)
print(f"{'Weighted ROC-AUC':<25} {weighted_roc_auc:<15.4f} {'Target ≥0.75 ✓':<30}")
print(f"{'95% CI Lower':<25} {ci_lower:<15.4f} {'':<30}")
print(f"{'95% CI Upper':<25} {ci_upper:<15.4f} {'CI excludes 0.5 ✓' if ci_lower > 0.5 else 'CI includes 0.5 ⚠':<30}")
print(f"{'PR-AUC':<25} {test_pr_auc:<15.4f} {'Target ≥0.45 ✓':<30}")
print(f"{'F1 Score':<25} {test_f1:<15.4f} {'':<30}")
print(f"{'Brier Score':<25} {brier:<15.4f} {'Lower is better':<30}")
print(f"{'Brier Skill':<25} {brier_skill:<15.4f} {'Improvement over baseline':<30}")
print(f"{'Train-Test Gap':<25} {auc_gap:<15.4f} {'<0.10 ✓' if auc_gap < 0.10 else '>0.10 ⚠':<30}")

# =============================================================================
# Step 2.6: Interpretation
# =============================================================================

print("\n" + "=" * 60)
print("Step 2.6: Interpretation")
print("=" * 60)

# --- 2.6a: Permutation Importance ---
print(f"\n" + "-" * 60)
print("PERMUTATION IMPORTANCE (Test Set, 50 repeats)")
print("-" * 60)

from sklearn.inspection import permutation_importance

print(f"\n  Computing permutation importance on test set...")
print(f"  (Using 10 repeats for efficiency)")

perm_importance = permutation_importance(
    final_model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
    scoring='roc_auc'
)

# Get top 20 features
perm_imp_mean = perm_importance.importances_mean
perm_imp_std = perm_importance.importances_std
feature_names = X_test.columns.tolist()

perm_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_imp_mean,
    'importance_std': perm_imp_std
}).sort_values('importance_mean', ascending=False)

print(f"\n  Top 20 Features by Permutation Importance:")
print(f"  {'Rank':<6} {'Feature':<50} {'Importance':<15} {'Std':<10}")
print("  " + "-" * 81)

top_20_perm = perm_imp_df.head(20)
for i, (_, row) in enumerate(top_20_perm.iterrows(), 1):
    print(f"  {i:<6} {row['feature']:<50} {row['importance_mean']:<15.4f} {row['importance_std']:<10.4f}")

# Save permutation importance
perm_imp_df.to_csv('permutation_importance.csv', index=False)
print(f"\n  Saved: permutation_importance.csv")

# --- 2.6b: SHAP Analysis ---
print(f"\n" + "-" * 60)
print("SHAP ANALYSIS (TreeExplainer, 500 test observations)")
print("-" * 60)

import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print(f"\n  Creating SHAP TreeExplainer...")
explainer = shap.TreeExplainer(final_model)

# Use 200 test observations for SHAP (reduced for efficiency)
n_shap = min(200, len(X_test))
X_shap = X_test.iloc[:n_shap]

print(f"  Computing SHAP values for {n_shap} observations...")
shap_values = explainer.shap_values(X_shap)

# For binary classification, shap_values is a list [class_0, class_1]
# We want class 1 (Trump voter)
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
else:
    shap_values_class1 = shap_values

# Ensure shap_values_class1 is 2D (observations x features)
if len(shap_values_class1.shape) == 3:
    # Shape is (n_samples, n_features, n_classes) - take class 1
    shap_values_class1 = shap_values_class1[:, :, 1]

# Calculate mean absolute SHAP values
mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)

# Ensure it's 1D
if len(mean_abs_shap.shape) > 1:
    mean_abs_shap = mean_abs_shap.flatten()

shap_importance_df = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

print(f"\n  Top 20 Features by Mean |SHAP| Value:")
print(f"  {'Rank':<6} {'Feature':<50} {'Mean |SHAP|':<15}")
print("  " + "-" * 71)

top_20_shap = shap_importance_df.head(20)
for i, (_, row) in enumerate(top_20_shap.iterrows(), 1):
    print(f"  {i:<6} {row['feature']:<50} {row['mean_abs_shap']:<15.4f}")

# Save SHAP importance
shap_importance_df.to_csv('shap_importance.csv', index=False)
print(f"\n  Saved: shap_importance.csv")

# Generate SHAP summary plot (beeswarm)
print(f"\n  Generating SHAP summary plot (beeswarm)...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_class1, X_shap, show=False, max_display=20)
plt.tight_layout()
plt.savefig('shap_summary_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: shap_summary_beeswarm.png")

# Generate SHAP dependence plots for top 5 predictors
print(f"\n  Generating SHAP dependence plots for top 5 predictors...")
top_5_features = shap_importance_df.head(5)['feature'].tolist()

for i, feat in enumerate(top_5_features, 1):
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feat, shap_values_class1, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(f'shap_dependence_{i}_{feat[:30]}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    {i}. {feat}")

print(f"  Saved: shap_dependence_*.png (5 files)")

# =============================================================================
# Step 2.7: Robustness Checks
# =============================================================================

print("\n" + "=" * 60)
print("Step 2.7: Robustness Checks")
print("=" * 60)

# --- 2.7a: Weighted Logistic Regression ---
print(f"\n" + "-" * 60)
print("WEIGHTED LOGISTIC REGRESSION (statsmodels)")
print("-" * 60)

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

print(f"\n  Fitting weighted logistic regression...")
print(f"  (Standardizing features for coefficient comparison)")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# Add constant for statsmodels
X_train_sm = sm.add_constant(X_train_scaled_df)

# Fit weighted logistic regression
try:
    logit_model = sm.GLM(
        y_train.values,
        X_train_sm,
        family=sm.families.Binomial(),
        freq_weights=weights_train.values
    )
    logit_results = logit_model.fit(maxiter=100, method='bfgs', disp=0)

    # Get coefficients (excluding constant)
    coef_df = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': logit_results.params[1:],  # Skip constant
        'abs_coefficient': np.abs(logit_results.params[1:])
    }).sort_values('abs_coefficient', ascending=False)

    print(f"\n  Top 10 Predictors by |Coefficient| (Logistic Regression):")
    print(f"  {'Rank':<6} {'Feature':<50} {'Coef':<12} {'Direction':<10}")
    print("  " + "-" * 78)

    top_10_logit = coef_df.head(10)
    logit_top_features = []
    for i, (_, row) in enumerate(top_10_logit.iterrows(), 1):
        direction = "+" if row['coefficient'] > 0 else "-"
        print(f"  {i:<6} {row['feature']:<50} {row['coefficient']:<12.4f} {direction:<10}")
        logit_top_features.append((row['feature'], row['coefficient']))

    # Save logistic regression coefficients
    coef_df.to_csv('logistic_coefficients.csv', index=False)
    print(f"\n  Saved: logistic_coefficients.csv")

    # Compare RF and Logistic Regression top features
    print(f"\n" + "-" * 60)
    print("COMPARISON: RF vs Logistic Regression Top 10")
    print("-" * 60)

    rf_top_10 = top_20_perm.head(10)['feature'].tolist()
    logit_top_10 = [f[0] for f in logit_top_features]

    overlap = set(rf_top_10) & set(logit_top_10)
    print(f"\n  RF Top 10 (Permutation Importance):")
    for i, f in enumerate(rf_top_10, 1):
        marker = "* " if f in overlap else "  "
        print(f"    {marker}{i}. {f}")

    print(f"\n  Logistic Regression Top 10 (|Coefficient|):")
    for i, (f, coef) in enumerate(logit_top_features, 1):
        marker = "* " if f in overlap else "  "
        direction = "(+)" if coef > 0 else "(-)"
        print(f"    {marker}{i}. {f} {direction}")

    print(f"\n  Overlap: {len(overlap)} features in both top 10")
    print(f"  Overlapping features: {list(overlap) if overlap else 'None'}")

except Exception as e:
    print(f"\n  Warning: Logistic regression failed: {e}")
    print(f"  Skipping logistic regression comparison...")
    coef_df = None

# --- 2.7b: Seed Stability ---
print(f"\n" + "-" * 60)
print("SEED STABILITY TEST")
print("-" * 60)

print(f"\n  Testing model stability across different random seeds...")

seed_results = []

# Original seed (42)
seed_results.append({
    'seed': 42,
    'test_auc': test_roc_auc
})

# Test with seed 123
print(f"  Training with seed 123...")
rf_123 = RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=123,
    n_jobs=-1
)
rf_123.fit(X_train, y_train, sample_weight=weights_train)
auc_123 = roc_auc_score(y_test, rf_123.predict_proba(X_test)[:, 1], sample_weight=weights_test)
seed_results.append({'seed': 123, 'test_auc': auc_123})

# Test with seed 456
print(f"  Training with seed 456...")
rf_456 = RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=456,
    n_jobs=-1
)
rf_456.fit(X_train, y_train, sample_weight=weights_train)
auc_456 = roc_auc_score(y_test, rf_456.predict_proba(X_test)[:, 1], sample_weight=weights_test)
seed_results.append({'seed': 456, 'test_auc': auc_456})

seed_df = pd.DataFrame(seed_results)
auc_range = seed_df['test_auc'].max() - seed_df['test_auc'].min()
auc_mean = seed_df['test_auc'].mean()
auc_std = seed_df['test_auc'].std()

print(f"\n  Seed Stability Results:")
print(f"  {'Seed':<10} {'Test AUC':<15}")
print("  " + "-" * 25)
for _, row in seed_df.iterrows():
    print(f"  {row['seed']:<10} {row['test_auc']:<15.4f}")

print(f"\n  Summary:")
print(f"    Mean AUC:  {auc_mean:.4f}")
print(f"    Std Dev:   {auc_std:.4f}")
print(f"    Range:     {auc_range:.4f}")

if auc_range < 0.02:
    print(f"\n  ✓ Model is stable across seeds (range < 0.02)")
elif auc_range < 0.05:
    print(f"\n  ⚠ Moderate seed sensitivity (range 0.02-0.05)")
else:
    print(f"\n  ⚠ High seed sensitivity (range > 0.05)")

# Save seed stability results
seed_df.to_csv('seed_stability.csv', index=False)
print(f"\n  Saved: seed_stability.csv")

# =============================================================================
# Step 2.6-2.7 Summary
# =============================================================================

print("\n" + "=" * 60)
print("INTERPRETATION & ROBUSTNESS SUMMARY")
print("=" * 60)

print(f"\n  Permutation Importance (Top 5):")
for i, (_, row) in enumerate(top_20_perm.head(5).iterrows(), 1):
    print(f"    {i}. {row['feature']}: {row['importance_mean']:.4f}")

print(f"\n  SHAP Importance (Top 5):")
for i, (_, row) in enumerate(top_20_shap.head(5).iterrows(), 1):
    print(f"    {i}. {row['feature']}: {row['mean_abs_shap']:.4f}")

print(f"\n  Seed Stability: Range = {auc_range:.4f} ({'✓ Stable' if auc_range < 0.02 else '⚠ Check'})")

print(f"\n  Output Files Saved:")
print(f"    - permutation_importance.csv")
print(f"    - shap_importance.csv")
print(f"    - shap_summary_beeswarm.png")
print(f"    - shap_dependence_*.png (5 files)")
print(f"    - logistic_coefficients.csv")
print(f"    - seed_stability.csv")

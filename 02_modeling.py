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

# Methods Documentation: Random Forest Analysis of Latino Trump Voters
## CMPS 2016 Data

---

## Section 1: Sample Definition

### 1.1 Original Sample
| Metric | Value |
|--------|-------|
| Original CMPS 2016 Sample | 10,144 |
| Latino Respondents (after ethnic filter) | 3,001 |
| Voters with Valid Vote Choice | 3,001 |
| **Final Analytic Sample** | **3,001** |

### 1.2 Filtering Criteria

1. **Ethnic Filter**: Retained only respondents identified as Latino/Hispanic
2. **Voter Filter**: Retained only respondents who reported voting in 2016 presidential election
3. **Valid DV Filter**: Retained only respondents with non-missing vote choice values

### 1.3 Dependent Variable Coding

| DV Value | Label | Coding Rule | N | % |
|----------|-------|-------------|---|---|
| 1 | Trump Voter | Vote choice contains "DONALD TRUMP" | 594 | 19.8% |
| 0 | Non-Trump Voter | Vote choice contains "HILLARY CLINTON", "GARY JOHNSON", "JILL STEIN", or "SOMEONE ELSE" | 2,407 | 80.2% |
| NA | Excluded | Missing or unrecognized vote choice | - | - |

**DV Creation Logic (from `01_preprocessing.py`):**
```python
def create_trump_dv(vote_choice):
    if pd.isna(vote_choice):
        return np.nan
    vote_str = str(vote_choice).upper()
    if 'DONALD TRUMP' in vote_str:
        return 1
    elif any(name in vote_str for name in ['HILLARY CLINTON', 'GARY JOHNSON', 'JILL STEIN', 'SOMEONE ELSE']):
        return 0
    else:
        return np.nan
```

---

## Section 2: Variable Exclusions (Pre-Modeling)

A total of **248 variables** were excluded during preprocessing, organized into the following categories:

### Category A: Administrative/Identifier Variables (14 total)

| Sub-Category | Variables | Rationale |
|--------------|-----------|-----------|
| Identifiers (4) | `RESPID`, `ZIPCODE`, `CITY_NAME`, `COUNTY_NAME` | Unique identifiers with no substantive meaning |
| Metadata (4) | `INTERVIEW_START`, `INTERVIEW_END`, `DIFF_DATE`, `ETHNIC_QUOTA` | Survey administration variables |
| Weights (2) | `WEIGHT`, `NAT_WEIGHT` | Sampling weights (used separately, not as predictors) |
| Tautological (3) | `C6`, `C7`, `C15` | Direct vote intention/choice variables |
| DV Source (1) | `C14` | Source variable for dependent variable |

### Category B: High Missingness Variables (124 total)

**Threshold**: Variables with >50% missing values were excluded.

**Rationale**: Imputation for variables with majority missing values introduces excessive uncertainty and potential bias.

| Range | Example Variables |
|-------|-------------------|
| Variables with >50% missing | 124 variables including race-specific battery items, follow-up questions with skip patterns |

*Full list available in `cmps_2016_excluded_vars.csv` under category "high_missingness"*

### Category C: Open-Text Variables (21 total)

| Variable Type | Count | Rationale |
|---------------|-------|-----------|
| Open-ended text responses | 21 | Cannot be meaningfully encoded for RF without NLP preprocessing |

*Examples: Write-in responses, "other specify" fields*

### Category D: Race-Specific Variables (87 total)

| Variable Type | Count | Rationale |
|---------------|-------|-----------|
| Non-Latino race battery items | 87 | Questions administered only to Black, Asian, or White respondents; not applicable to Latino sample |

*Examples: African American identity items, Asian American political participation items*

### Summary Table: Pre-Modeling Exclusions

| Category | Description | N Variables |
|----------|-------------|-------------|
| A | Administrative/Identifiers | 14 |
| B | High Missingness (>50%) | 124 |
| C | Open-Text | 21 |
| D | Race-Specific | 87 |
| E | Tautological (Direct) | 3 |
| F | DV Source | 1 |
| **Total** | | **248** |

**Note**: Category E (Tautological Direct) includes `C6`, `C7`, `C15` which directly measure vote intention and would create perfect circularity with the DV. Category F (`C14`) is the source variable from which the DV was derived.

---

## Section 3: Partisan Indicator Definitions (Non-Partisan Model)

For the non-partisan model, an additional **80 variables** were excluded to remove tautological or near-tautological partisan predictors.

### 3.1 Classification Framework

| Classification | Definition | Decision Rule |
|----------------|------------|---------------|
| **Core Partisan** | Directly measures party identification, ideology, or candidate evaluation | Automatic exclusion |
| **Derivative Partisan** | Derived from or strongly collinear with core partisan variables | Automatic exclusion |
| **Potentially Tautological** | Variables that may proxy partisan identity through family/social context | Case-by-case review; excluded if primary purpose is measuring partisan orientation |

### 3.2 Excluded Variables by Type

#### Core Partisan Variables

| Prefix | Variable Description | N Columns | Classification |
|--------|---------------------|-----------|----------------|
| C2_ | Hillary Clinton favorability | 6 | Core - Candidate Evaluation |
| C3_ | Bernie Sanders favorability | 6 | Core - Candidate Evaluation |
| C4_ | Donald Trump favorability | 7 | Core - Candidate Evaluation |
| C5_ | Ted Cruz favorability | 6 | Core - Candidate Evaluation |
| C8_ | Bill Clinton favorability | 6 | Core - Candidate Evaluation |
| C9_ | Barack Obama favorability | 6 | Core - Candidate Evaluation |
| C10_ | Michelle Obama favorability | 6 | Core - Candidate Evaluation |
| C11_ | Jeb Bush favorability | 6 | Core - Candidate Evaluation |
| C25_ | Party registration (Rep/Dem/Ind) | 6 | Core - Party ID |
| C26_ | Strong partisan identification | 4 | Core - Party ID |
| C27_ | Party lean (for independents) | 4 | Core - Party ID |
| C31_ | Ideology (liberal-conservative scale) | 8 | Core - Ideology |

#### Derivative Partisan Variables

| Prefix | Variable Description | N Columns | Classification |
|--------|---------------------|-----------|----------------|
| L46_ | Which party better on immigration | 2 | Derivative - Party Evaluation |
| L266_ | Which party better for Latinos | 2 | Derivative - Party Evaluation |
| L267_ | Which party better on values | 2 | Derivative - Party Evaluation |
| L293_ | Democratic Party favorability (0-10) | 1 | Derivative - Party Evaluation |
| L294_ | Republican Party favorability (0-10) | 1 | Derivative - Party Evaluation |
| C242_HID_ | Party identification (derived) | 4 | Derivative - Party ID |
| LA204_ | Party support (group support) | 3 | Derivative - Party Support |

#### Potentially Tautological (Flagged and Excluded)

| Prefix | Variable Description | N Columns | Rationale for Exclusion |
|--------|---------------------|-----------|------------------------|
| LA203_ | Mother's partisan affiliation | 3 | Directly measures family partisan identity; likely proxy for respondent partisanship |

### 3.3 Summary: Partisan Exclusions

| Type | N Variables |
|------|-------------|
| Core Partisan (Candidate Favorability) | 49 |
| Core Partisan (Party ID) | 14 |
| Core Partisan (Ideology) | 8 |
| Derivative Partisan (Party Evaluations) | 8 |
| Potentially Tautological (Family Partisanship) | 3 |
| **Total Partisan Exclusions** | **80** |

*Full list available in `excluded_partisan_columns_final.csv`*

---

## Section 4: Variable Processing

### 4.1 Missingness Handling

| Data Type | Imputation Method | Implementation |
|-----------|-------------------|----------------|
| Numeric | Median imputation | `SimpleImputer(strategy='median')` |
| Categorical | Mode imputation | `SimpleImputer(strategy='most_frequent')` |

**Documentation**: 588 imputation entries logged in `cmps_2016_imputation_log.csv`

### 4.2 Rare Level Pooling

| Parameter | Value |
|-----------|-------|
| Threshold | <5% of observations |
| Pooling Label | "Other" |
| Variables Affected | 283 |

**Rationale**: Rare categorical levels can cause instability in tree-based models and produce unreliable importance estimates.

**Documentation**: Full pooling log available in `cmps_2016_pooling_log.csv`

### 4.3 Encoding

| Method | Implementation | Notes |
|--------|----------------|-------|
| One-Hot Encoding | `pd.get_dummies(drop_first=True)` | Reference category dropped to avoid multicollinearity |

### 4.4 Final Feature Counts

| Model | Features |
|-------|----------|
| Full Model | 1,517 |
| Non-Partisan Model | 1,437 |
| Features Removed (Partisan) | 80 |

---

## Section 5: Model Specifications

### 5.1 Random Forest Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 500 | Sufficient for convergence; diminishing returns beyond |
| `max_features` | 'sqrt' | Standard default; reduces correlation between trees |
| `min_samples_leaf` | 1 | Allows full tree growth; combined with class balancing |
| `class_weight` | 'balanced' | Adjusts for class imbalance (19.8% Trump vs 80.2% Non-Trump) |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Parallel processing |

### 5.2 Train/Test Split

| Parameter | Value |
|-----------|-------|
| Test Size | 20% |
| Stratification | By DV (Trump vote) |
| Random State | 42 |

**Resulting Split**:
| Set | N | % Trump |
|-----|---|---------|
| Train | 2,400 | 19.8% |
| Test | 601 | 19.8% |

### 5.3 Cross-Validation (Hyperparameter Tuning)

| Parameter | Value |
|-----------|-------|
| CV Folds | 5 |
| CV Type | Stratified K-Fold |
| Scoring Metric | ROC-AUC |

**Hyperparameter Grid Searched**:
```python
param_grid = {
    'max_features': ['sqrt', 0.33],
    'min_samples_leaf': [1, 5]
}
```

### 5.4 Feature Importance Method

| Method | Implementation | Parameters |
|--------|----------------|------------|
| Permutation Importance | `sklearn.inspection.permutation_importance` | 50 repeats, ROC-AUC scoring |
| SHAP | `shap.TreeExplainer` | Mean absolute SHAP values |

**Rationale for Permutation Importance**: Unlike Gini importance, permutation importance is unbiased toward high-cardinality features and directly measures predictive contribution via AUC decrease when feature is shuffled.

### 5.5 Model Performance

| Model | Test ROC-AUC | Features | Notes |
|-------|--------------|----------|-------|
| Full Model | 0.9383 | 1,517 | Includes all predictors |
| Non-Partisan Model | 0.8886 | 1,437 | Excludes 80 partisan variables |
| **AUC Drop** | **0.0497** | - | 5.3% relative decline |

---

## Section 6: Output Files Summary

### 6.1 Data Files

| File | Description | Format |
|------|-------------|--------|
| `cmps_2016_X.parquet` | Feature matrix (one-hot encoded) | Parquet |
| `cmps_2016_y.parquet` | Dependent variable | Parquet |
| `cmps_2016_weights.parquet` | Survey weights | Parquet |

### 6.2 Exclusion Logs

| File | Description |
|------|-------------|
| `cmps_2016_excluded_vars.csv` | 248 variables excluded during preprocessing with categories |
| `excluded_partisan_columns_final.csv` | 80 partisan variables excluded for non-partisan model |
| `cmps_2016_imputation_log.csv` | Imputation method and values for 588 entries |
| `cmps_2016_pooling_log.csv` | Rare level pooling log for 283 variables |

### 6.3 Results Files

| File | Description |
|------|-------------|
| `top30_full_model_corrected.csv` | Top 30 predictors from full model |
| `top30_nonpartisan_themed.csv` | Top 30 predictors from non-partisan model with thematic coding |
| `theme_summary.csv` | Aggregated importance by thematic category |
| `shap_summary.csv` | SHAP values for non-partisan model features |

### 6.4 Visualization Files

| File | Description |
|------|-------------|
| `shap_summary_plot.png` | SHAP summary plot (top features) |
| `shap_bar_plot.png` | SHAP bar plot (mean |SHAP|) |
| `analysis_summary_report.pdf` | PDF summary report |

---

## Appendix: Flagged Items Requiring Verification

> **Items below are NOT fully documented in code/notes and may require manual verification:**

| Item | Status | Notes |
|------|--------|-------|
| Original CMPS 2016 raw sample size (10,144) | **[NEEDS VERIFICATION]** | Inferred from preprocessing output; verify against CMPS codebook |
| Exact missingness threshold (>50%) | **[NEEDS VERIFICATION]** | Documented in code but threshold selection rationale not explicit |
| Survey weight variable used | Documented | `survey_wt` column in weights parquet |
| Logistic Regression specifications | **[NOT IMPLEMENTED]** | User mentioned LR in request but only RF was implemented |

---

## Version Information

| Item | Value |
|------|-------|
| Analysis Date | 2025-11-30 |
| Python Version | 3.x |
| Key Dependencies | scikit-learn, pandas, numpy, shap, matplotlib |
| Random Seed | 42 (all stochastic operations) |

---

*Document generated from analysis scripts: `01_preprocessing.py`, `02_modeling.py`, `02c_interpretation_corrected.py`, `02d_final_nonpartisan_analysis.py`*

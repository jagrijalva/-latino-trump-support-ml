# ============================================================
# CMPS 2016 Random Forest Preparation Script
# Prepares Latino subsample for Trump vote prediction
# Uses base R only (no external dependencies)
# ============================================================

# Load the data
load("CMPS_2016_raw.rda")
df <- da38040.0001

cat(strrep("=", 60), "\n")
cat("CMPS 2016 RANDOM FOREST DATA PREPARATION\n")
cat(strrep("=", 60), "\n\n")

# ============================================================
# STEP 1: FILTER TO LATINO RESPONDENTS
# ============================================================
cat("STEP 1: FILTER TO LATINO RESPONDENTS\n")
cat(strrep("-", 40), "\n")

latinos <- df[df$ETHNIC_QUOTA == "(2) Hispanic or Latino", ]
cat("Original sample size:", nrow(df), "\n")
cat("Latino sample size:", nrow(latinos), "\n\n")

# ============================================================
# STEP 2: CREATE BINARY DEPENDENT VARIABLE
# ============================================================
cat("STEP 2: CREATE BINARY DEPENDENT VARIABLE\n")
cat(strrep("-", 40), "\n")

# Check C14 distribution for Latinos
cat("C14 (Vote Choice) distribution for Latinos:\n")
print(table(latinos$C14, useNA = "always"))

# Binary DV: Trump (1) vs All Other Voters (0)
# Include: Clinton, Johnson, Stein, Someone else
# Exclude: Non-voters (NA in C14)

# Filter to respondents with valid vote choice (exclude NA)
latinos_voters <- latinos[!is.na(latinos$C14), ]

# Create binary DV: Trump = 1, All others = 0
latinos_voters$trump_vote <- ifelse(latinos_voters$C14 == "(2) Donald Trump", 1, 0)

cat("\nBinary DV: Trump (1) vs All Other Voters (0)\n")
cat("  Total voters:", nrow(latinos_voters), "\n")
cat("  Trump votes:", sum(latinos_voters$trump_vote), "\n")
cat("  Non-Trump votes:", sum(latinos_voters$trump_vote == 0), "\n")
cat("  Trump %:", round(100 * mean(latinos_voters$trump_vote), 2), "%\n")

cat("\nNon-Trump breakdown:\n")
non_trump <- latinos_voters[latinos_voters$trump_vote == 0, ]
print(table(non_trump$C14))

# ============================================================
# STEP 3: IDENTIFY VARIABLES TO EXCLUDE
# ============================================================
cat("\n\nSTEP 3: IDENTIFY VARIABLES TO EXCLUDE\n")
cat(strrep("-", 40), "\n")

all_vars <- names(latinos)

# 3a. Identifiers and metadata
exclude_identifiers <- c("RESPID", "INTERVIEW_START", "INTERVIEW_END", "DIFF_DATE",
                         "ZIPCODE", "CITY_NAME", "COUNTY_NAME")

# 3b. Dependent variable and weights
exclude_dv_weights <- c("C14", "WEIGHT", "NAT_WEIGHT")

# 3c. Variables with race-specific prefixes (not asked of Latinos)
# Check missingness for all variables starting with A, B, BW, BA in Latino subsample
potential_race_specific <- all_vars[grepl("^A[0-9]|^A[0-9][0-9]|^B[0-9]|^B[0-9][0-9]|^BW|^BA[0-9]", all_vars)]

cat("\nChecking", length(potential_race_specific), "potential race-specific variables...\n")

high_missing_vars <- c()
for (v in potential_race_specific) {
  n_missing <- sum(is.na(latinos[[v]]))
  pct_missing <- 100 * n_missing / nrow(latinos)
  if (pct_missing > 95) {
    high_missing_vars <- c(high_missing_vars, v)
  }
}

cat("Variables with >95% missing for Latinos:", length(high_missing_vars), "\n")

# 3d. Split indicators
exclude_splits <- all_vars[grepl("^SPLIT", all_vars)]
cat("Split indicator variables:", length(exclude_splits), "\n")

# 3e. Other metadata to exclude
exclude_other <- c("FLAG_DESCRIPTION", "ETHNIC_QUOTA", "RESPONSE_MINUTES")

# 3f. Variables that are Latino-specific (L*, LA*) - KEEP THESE
latino_specific <- all_vars[grepl("^L[0-9]|^LA[0-9]", all_vars)]
cat("Latino-specific variables (L*, LA*) to KEEP:", length(latino_specific), "\n")

# 3g. Common questions (C*) - KEEP (except C14)
common_vars <- all_vars[grepl("^C[0-9]", all_vars)]
common_vars <- setdiff(common_vars, "C14")
cat("Common question variables (C*) to KEEP:", length(common_vars), "\n")

# 3h. Census/context variables (DP*, EC*, SC*, HC*, HISP*) - KEEP
census_vars <- all_vars[grepl("^DP|^EC|^SC|^HC|^HISP[0-9]", all_vars)]
cat("Census/context variables to KEEP:", length(census_vars), "\n")

# ============================================================
# COMPILE FINAL EXCLUSION LIST
# ============================================================
cat("\n\nFINAL VARIABLE EXCLUSION LIST\n")
cat(strrep("-", 40), "\n")

exclude_all <- unique(c(
  exclude_identifiers,
  exclude_dv_weights,
  high_missing_vars,
  exclude_splits,
  exclude_other
))

cat("Total variables to exclude:", length(exclude_all), "\n\n")

cat("Exclusion breakdown:\n")
cat("  Identifiers/metadata:      ", length(exclude_identifiers), "\n")
cat("  DV and weights:            ", length(exclude_dv_weights), "\n")
cat("  High-missing race-specific:", length(high_missing_vars), "\n")
cat("  Split indicators:          ", length(exclude_splits), "\n")
cat("  Other metadata:            ", length(exclude_other), "\n")

# List high-missing variables
if (length(high_missing_vars) > 0) {
  cat("\nHigh-missing race-specific variables (first 30):\n")
  for (v in head(high_missing_vars, 30)) {
    cat("  ", v, "\n")
  }
  if (length(high_missing_vars) > 30) {
    cat("  ... and", length(high_missing_vars) - 30, "more\n")
  }
}

# ============================================================
# CREATE FINAL RF DATASET
# ============================================================
cat("\n\nSTEP 4: CREATE FINAL RF DATASET\n")
cat(strrep("-", 40), "\n")

# Use all Latino voters with Trump vs All Others
rf_data <- latinos_voters

# Remove excluded variables
vars_to_remove <- exclude_all[exclude_all %in% names(rf_data)]
rf_data <- rf_data[, !(names(rf_data) %in% vars_to_remove)]

cat("Final RF dataset dimensions:", nrow(rf_data), "x", ncol(rf_data), "\n")

# Check for remaining variables with high missingness
missing_counts <- sapply(rf_data, function(x) sum(is.na(x)))
high_missing_final <- names(missing_counts[missing_counts > 0.5 * nrow(rf_data)])

cat("Variables with >50% missing in final dataset:", length(high_missing_final), "\n")

if (length(high_missing_final) > 0) {
  cat("\nThese should be reviewed or excluded:\n")
  for (v in head(high_missing_final, 30)) {
    cat("  ", v, ":", missing_counts[v], "missing (",
        round(100*missing_counts[v]/nrow(rf_data), 1), "%)\n")
  }
}

# ============================================================
# ADDITIONAL CLEANUP: Remove remaining high-missing variables
# ============================================================
cat("\n\nSTEP 5: ADDITIONAL CLEANUP\n")
cat(strrep("-", 40), "\n")

# Remove variables with >50% missing from final dataset
if (length(high_missing_final) > 0) {
  rf_data_clean <- rf_data[, !(names(rf_data) %in% high_missing_final)]
  cat("Removed", length(high_missing_final), "additional high-missing variables\n")
  cat("Clean RF dataset dimensions:", nrow(rf_data_clean), "x", ncol(rf_data_clean), "\n")
} else {
  rf_data_clean <- rf_data
  cat("No additional variables removed\n")
}

# ============================================================
# SAVE OUTPUTS
# ============================================================
cat("\n\nSAVING OUTPUTS\n")
cat(strrep("-", 40), "\n")

# Save the RF-ready dataset (with high-missing vars)
save(rf_data, file = "cmps_2016_latino_rf_data.rda")
cat("Saved: cmps_2016_latino_rf_data.rda (includes vars with some missing)\n")

# Save clean version
save(rf_data_clean, file = "cmps_2016_latino_rf_data_clean.rda")
cat("Saved: cmps_2016_latino_rf_data_clean.rda (removed >50% missing)\n")

# Save variable exclusion documentation
exclusion_doc <- data.frame(
  variable = exclude_all,
  reason = sapply(exclude_all, function(v) {
    if (v %in% exclude_identifiers) return("Identifier/metadata")
    if (v %in% exclude_dv_weights) return("DV or weight variable")
    if (v %in% high_missing_vars) return("High missing (>95%) - race-specific")
    if (v %in% exclude_splits) return("Split indicator")
    if (v %in% exclude_other) return("Other metadata")
    return("Unknown")
  }),
  stringsAsFactors = FALSE
)
write.csv(exclusion_doc, "cmps_2016_excluded_variables.csv", row.names = FALSE)
cat("Saved: cmps_2016_excluded_variables.csv\n")

# Save list of retained variables (clean version)
retained_vars <- data.frame(
  variable = setdiff(names(rf_data_clean), "trump_vote"),
  type = sapply(rf_data_clean[, setdiff(names(rf_data_clean), "trump_vote"), drop=FALSE],
                function(x) class(x)[1]),
  n_missing = sapply(rf_data_clean[, setdiff(names(rf_data_clean), "trump_vote"), drop=FALSE],
                     function(x) sum(is.na(x))),
  stringsAsFactors = FALSE
)
retained_vars$pct_missing <- round(100 * retained_vars$n_missing / nrow(rf_data_clean), 2)
write.csv(retained_vars, "cmps_2016_retained_variables.csv", row.names = FALSE)
cat("Saved: cmps_2016_retained_variables.csv\n")

# ============================================================
# FINAL SUMMARY
# ============================================================
cat("\n\n")
cat(strrep("=", 60), "\n")
cat("RF PREPARATION SUMMARY\n")
cat(strrep("=", 60), "\n")
cat("Original dataset:       ", nrow(df), "x", ncol(df), "\n")
cat("Latino subsample:       ", nrow(latinos), "respondents\n")
cat("Latino voters:          ", nrow(latinos_voters), "(excludes non-voters)\n")
cat("Final RF dataset:       ", nrow(rf_data_clean), "x", ncol(rf_data_clean), "\n")
cat("\n")
cat("DV: trump_vote (Trump=1 vs All Others=0)\n")
cat("  Trump (1):            ", sum(rf_data_clean$trump_vote), "\n")
cat("  Non-Trump (0):        ", sum(rf_data_clean$trump_vote == 0), "\n")
cat("  Class balance:        ", round(100*mean(rf_data_clean$trump_vote), 2), "% Trump\n")
cat("\n")
cat("Variables excluded:     ", length(exclude_all) + length(high_missing_final), "\n")
cat("Variables retained:     ", ncol(rf_data_clean) - 1, "(plus trump_vote DV)\n")
cat(strrep("=", 60), "\n")

# ============================================================
# CMPS 2016 Missingness Diagnostic for Latino RF Dataset
# Analyzes missingness patterns in the filtered Latino subset
# ============================================================

# Load the RF-ready data (before >50% removal)
load("cmps_2016_latino_rf_data.rda")

cat(strrep("=", 70), "\n")
cat("MISSINGNESS DIAGNOSTIC FOR LATINO RF DATASET\n")
cat(strrep("=", 70), "\n\n")

cat("Dataset dimensions:", nrow(rf_data), "x", ncol(rf_data), "\n\n")

# ============================================================
# CALCULATE MISSINGNESS FOR ALL PREDICTORS
# ============================================================

# Exclude the DV (trump_vote) from analysis
predictors <- setdiff(names(rf_data), "trump_vote")

# Calculate missingness stats for each predictor
missingness_df <- data.frame(
  variable = predictors,
  type = sapply(rf_data[, predictors, drop=FALSE], function(x) class(x)[1]),
  n_missing = sapply(rf_data[, predictors, drop=FALSE], function(x) sum(is.na(x))),
  stringsAsFactors = FALSE
)
missingness_df$pct_missing <- round(100 * missingness_df$n_missing / nrow(rf_data), 2)

# Sort by missingness (highest first)
missingness_df <- missingness_df[order(-missingness_df$pct_missing), ]

# ============================================================
# DISTRIBUTION OF MISSINGNESS
# ============================================================
cat(strrep("-", 70), "\n")
cat("DISTRIBUTION OF MISSINGNESS\n")
cat(strrep("-", 70), "\n\n")

# Create missingness bins
missingness_df$bin <- cut(missingness_df$pct_missing,
                          breaks = c(-0.1, 0, 10, 30, 50, 100),
                          labels = c("0% (complete)", "<10%", "10-30%", "30-50%", ">50%"))

# Count by bin
bin_counts <- table(missingness_df$bin)
bin_pct <- round(100 * bin_counts / sum(bin_counts), 1)

cat("Missingness Category     Count    Percent\n")
cat(strrep("-", 45), "\n")
for (i in 1:length(bin_counts)) {
  cat(sprintf("%-20s %8d %10.1f%%\n", names(bin_counts)[i], bin_counts[i], bin_pct[i]))
}
cat(strrep("-", 45), "\n")
cat(sprintf("%-20s %8d %10.1f%%\n", "TOTAL", sum(bin_counts), 100))

# By variable type
cat("\n\nMissingness by Variable Type:\n")
cat(strrep("-", 45), "\n")
type_summary <- aggregate(cbind(n_missing, pct_missing) ~ type, data = missingness_df,
                          FUN = function(x) c(count = length(x), mean = round(mean(x), 2)))
for (t in unique(missingness_df$type)) {
  type_subset <- missingness_df[missingness_df$type == t, ]
  cat(sprintf("\n%s variables: %d\n", t, nrow(type_subset)))
  cat(sprintf("  Mean missingness: %.2f%%\n", mean(type_subset$pct_missing)))
  cat(sprintf("  Complete (0%% missing): %d\n", sum(type_subset$pct_missing == 0)))
  cat(sprintf("  >50%% missing: %d\n", sum(type_subset$pct_missing > 50)))
}

# ============================================================
# VARIABLES WITH >50% MISSING (CANDIDATES FOR REMOVAL)
# ============================================================
cat("\n\n")
cat(strrep("-", 70), "\n")
cat("VARIABLES WITH >50% MISSING (CANDIDATES FOR REMOVAL)\n")
cat(strrep("-", 70), "\n\n")

high_missing <- missingness_df[missingness_df$pct_missing > 50, ]
cat("Total variables >50% missing:", nrow(high_missing), "\n\n")

if (nrow(high_missing) > 0) {
  cat(sprintf("%-40s %-10s %8s %10s\n", "Variable", "Type", "N_Miss", "Pct_Miss"))
  cat(strrep("-", 70), "\n")
  for (i in 1:nrow(high_missing)) {
    cat(sprintf("%-40s %-10s %8d %9.1f%%\n",
                substr(high_missing$variable[i], 1, 40),
                high_missing$type[i],
                high_missing$n_missing[i],
                high_missing$pct_missing[i]))
  }
}

# ============================================================
# VARIABLES WITH 0% MISSING (CLEANEST PREDICTORS)
# ============================================================
cat("\n\n")
cat(strrep("-", 70), "\n")
cat("VARIABLES WITH 0% MISSING (CLEANEST PREDICTORS)\n")
cat(strrep("-", 70), "\n\n")

complete_vars <- missingness_df[missingness_df$pct_missing == 0, ]
cat("Total variables with 0% missing:", nrow(complete_vars), "\n\n")

# Group by type
complete_by_type <- table(complete_vars$type)
cat("By type:\n")
for (i in 1:length(complete_by_type)) {
  cat(sprintf("  %s: %d\n", names(complete_by_type)[i], complete_by_type[i]))
}

cat("\nComplete variable list:\n")
cat(strrep("-", 70), "\n")
cat(sprintf("%-45s %-10s\n", "Variable", "Type"))
cat(strrep("-", 70), "\n")

# Sort alphabetically for easier reading
complete_vars_sorted <- complete_vars[order(complete_vars$variable), ]
for (i in 1:nrow(complete_vars_sorted)) {
  cat(sprintf("%-45s %-10s\n",
              substr(complete_vars_sorted$variable[i], 1, 45),
              complete_vars_sorted$type[i]))
}

# ============================================================
# VARIABLES WITH <10% MISSING (GOOD PREDICTORS)
# ============================================================
cat("\n\n")
cat(strrep("-", 70), "\n")
cat("VARIABLES WITH 1-10% MISSING (GOOD PREDICTORS)\n")
cat(strrep("-", 70), "\n\n")

good_vars <- missingness_df[missingness_df$pct_missing > 0 & missingness_df$pct_missing <= 10, ]
cat("Total variables with 1-10% missing:", nrow(good_vars), "\n\n")

if (nrow(good_vars) > 0) {
  cat(sprintf("%-40s %-10s %8s %10s\n", "Variable", "Type", "N_Miss", "Pct_Miss"))
  cat(strrep("-", 70), "\n")
  good_vars_sorted <- good_vars[order(good_vars$pct_missing), ]
  for (i in 1:nrow(good_vars_sorted)) {
    cat(sprintf("%-40s %-10s %8d %9.2f%%\n",
                substr(good_vars_sorted$variable[i], 1, 40),
                good_vars_sorted$type[i],
                good_vars_sorted$n_missing[i],
                good_vars_sorted$pct_missing[i]))
  }
}

# ============================================================
# SAVE FULL MISSINGNESS REPORT
# ============================================================
cat("\n\n")
cat(strrep("-", 70), "\n")
cat("SAVING OUTPUTS\n")
cat(strrep("-", 70), "\n\n")

# Save full missingness report
write.csv(missingness_df, "cmps_2016_latino_missingness_report.csv", row.names = FALSE)
cat("Saved: cmps_2016_latino_missingness_report.csv\n")

# Save list of clean variables (0% missing)
write.csv(complete_vars[, c("variable", "type")], "cmps_2016_latino_complete_vars.csv", row.names = FALSE)
cat("Saved: cmps_2016_latino_complete_vars.csv\n")

# Save list of high-missing variables
write.csv(high_missing, "cmps_2016_latino_high_missing_vars.csv", row.names = FALSE)
cat("Saved: cmps_2016_latino_high_missing_vars.csv\n")

# ============================================================
# SUMMARY
# ============================================================
cat("\n\n")
cat(strrep("=", 70), "\n")
cat("MISSINGNESS SUMMARY\n")
cat(strrep("=", 70), "\n")
cat("Total predictors analyzed:", length(predictors), "\n")
cat("Complete (0% missing):    ", nrow(complete_vars), "(", round(100*nrow(complete_vars)/length(predictors), 1), "%)\n")
cat("<10% missing:             ", sum(missingness_df$pct_missing < 10), "(", round(100*sum(missingness_df$pct_missing < 10)/length(predictors), 1), "%)\n")
cat(">50% missing (remove):    ", nrow(high_missing), "(", round(100*nrow(high_missing)/length(predictors), 1), "%)\n")
cat(strrep("=", 70), "\n")

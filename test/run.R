cat("\n═══════════════════════════════════════════════════════════\n")
cat("  QuantMgmt Spark Classifier Tests\n")
cat("═══════════════════════════════════════════════════════════\n\n")

# Step 1: Source R files from GitHub
cat("Step 1: Sourcing QuantMgmt R files from GitHub...\n")

github_base <- "https://raw.githubusercontent.com/laixx214/QuantMgmt/refs/heads/main/R"
r_files <- c(
  "constants.R",
  "validation_utils.R",
  "binary_classification_metrics.R",
  "auto_tune_classifier.R",
  "auto_tune_classifier_spark.R",
  "predict_classifier.R",
  "evaluate_classifier_performance.R"
)

for (file in r_files) {
  url <- paste0(github_base, "/", file)
  cat("  Sourcing:", file, "\n")
  source(url)
}
cat("  ✅ All R files sourced from GitHub\n\n")

cat("Loading required packages...\n")
library(sparklyr)
library(dplyr)
cat("  ✅ Packages loaded\n\n")

# Step 2: Get Spark connection
cat("Step 2: Finding Spark connection...\n")
sc <- spark_connect(method = "databricks")
DBI::dbExecute(sc, "USE home_yufeng_lai.default")
cat("  ✅ Connected:", class(sc)[1], "\n\n")

# Step 3: Prepare data
cat("Step 3: Preparing test data...\n")
data(iris)
set.seed(123)

iris_binary <- iris
iris_binary$is_setosa <- ifelse(iris_binary$Species == "setosa", 1, 0)
iris_binary$Species <- NULL

train_idx <- sample(1:nrow(iris_binary), 0.7 * nrow(iris_binary))
X_train <- iris_binary[train_idx, 1:4]
Y_train <- iris_binary[train_idx, "is_setosa"]
X_val <- iris_binary[-train_idx, 1:4]
Y_val <- iris_binary[-train_idx, "is_setosa"]

cat("  ✅ Iris data: ", nrow(X_train), " train, ", nrow(X_val), " val\n\n", sep = "")

# Step 4: Configure algorithms
cat("Step 4: Configuring algorithms...\n")
algorithms <- list(
  rf = list(
    learner = "random_forest",
    param_space = list(num_trees = c(50, 100), max_depth = c(3, 5)),
    measure = "auc"
  )
)
cat("  ✅ Random Forest: 2x2 param grid, 2-fold CV, 2 evals\n\n")

# Step 5: Test sequential training
cat("═══════════════════════════════════════════════════════════\n")
cat("Step 5: Sequential Training (parallelism=1)\n")
cat("═══════════════════════════════════════════════════════════\n\n")

t1 <- Sys.time()
res_seq <- auto_tune_classifier_spark(
  sc, X_train, Y_train, NULL, algorithms, 2, 2, "all", 123, 1, TRUE
)
time_seq <- as.numeric(difftime(Sys.time(), t1, units = "secs"))

cat("\n  ✅ Sequential: ", round(time_seq, 2), " sec, ",
    length(res_seq$tuned) + length(res_seq$untuned), " models\n\n", sep = "")

# Step 6: Test parallel training
cat("═══════════════════════════════════════════════════════════\n")
cat("Step 6: Parallel Training (parallelism=4)\n")
cat("═══════════════════════════════════════════════════════════\n\n")

t2 <- Sys.time()
res_par <- auto_tune_classifier_spark(
  sc, X_train, Y_train, NULL, algorithms, 2, 2, "all", 123, 4, TRUE
)
time_par <- as.numeric(difftime(Sys.time(), t2, units = "secs"))

cat("\n  ✅ Parallel: ", round(time_par, 2), " sec, speedup: ",
    round(time_seq/time_par, 2), "x\n\n", sep = "")

# Step 7: Test predictions
cat("═══════════════════════════════════════════════════════════\n")
cat("Step 7: Testing Predictions\n")
cat("═══════════════════════════════════════════════════════════\n\n")

preds <- predict_classifier(res_par, X_val)

# Check dimensions
dim_ok <- all(sapply(preds$tuned_prediction, length) == nrow(X_val))

cat("  ✅ Predictions: tuned=", length(preds$tuned_prediction),
    ", untuned=", length(preds$untuned_prediction), "\n", sep = "")
cat("  ", ifelse(dim_ok, "✅", "❌"), " Dimensions correct\n\n", sep = "")

# Step 8: Test evaluation
cat("═══════════════════════════════════════════════════════════\n")
cat("Step 8: Testing Evaluation\n")
cat("═══════════════════════════════════════════════════════════\n\n")

perf <- evaluate_classifier_performance(
  res_par, list(X_validate = X_val, Y_validate = Y_val), 0.5
)

cat("  ✅ Performance metrics calculated\n\n")
print(perf[, c("algorithm", "model_type", "classif.auc", "classif.acc")])

# Step 9: Summary and bug detection
cat("\n\n═══════════════════════════════════════════════════════════\n")
cat("  TEST SUMMARY\n")
cat("═══════════════════════════════════════════════════════════\n\n")

# Bug checks
bugs <- c()
if (!dim_ok) bugs <- c(bugs, "❌ Prediction dimensions incorrect")
if (any(is.na(perf$classif.auc))) bugs <- c(bugs, "❌ NA values in metrics")
if (any(perf$classif.auc < 0 | perf$classif.auc > 1)) bugs <- c(bugs, "❌ Invalid AUC range")
if (time_par >= time_seq * 1.5) bugs <- c(bugs, "⚠️  Parallelism not effective")

cat("Results:\n")
cat("  Sequential:", round(time_seq, 2), "sec\n")
cat("  Parallel:", round(time_par, 2), "sec (", round(time_seq/time_par, 2), "x speedup)\n", sep = "")
cat("  Best AUC:", round(max(perf[perf$model_type=="tuned", "classif.auc"]), 4), "\n")
cat("  Metrics valid:", all(perf$classif.auc >= 0 & perf$classif.auc <= 1), "\n")
cat("  Predictions OK:", dim_ok, "\n\n")

if (length(bugs) > 0) {
  cat("BUGS DETECTED:\n")
  for (bug in bugs) cat("  ", bug, "\n", sep = "")
  cat("\n")
} else {
  cat("✅ NO BUGS DETECTED - ALL TESTS PASSED!\n\n")
}

cat("═══════════════════════════════════════════════════════════\n\n")

invisible(list(
  sequential_time = time_seq,
  parallel_time = time_par,
  speedup = time_seq / time_par,
  performance = perf,
  predictions = preds,
  bugs = bugs
))
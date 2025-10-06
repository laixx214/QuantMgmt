# Databricks notebook source
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

# COMMAND ----------
# Step 2: Get Spark connection
cat("Step 2: Finding Spark connection...\n")
sc <- spark_connect(method = "databricks")
DBI::dbExecute(sc, "USE home_yufeng_lai.default")
cat("  ✅ Connected:", class(sc)[1], "\n\n")

# COMMAND ----------
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

# Prepare combined dataframe for data parameter tests
train_data <- iris_binary[train_idx, ]
val_data <- iris_binary[-train_idx, ]

cat("  ✅ Iris data: ", nrow(X_train), " train, ", nrow(X_val), " val\n\n", sep = "")

# COMMAND ----------
# Step 4: Configure algorithms
cat("Step 4: Configuring algorithms...\n")
algorithms <- list(
  rf = list(
    learner = "random_forest",
    param_space = list(num_trees = c(50, 100), max_depth = c(3, 5)),
    measure = "areaUnderPR"
  )
)
cat("  ✅ Random Forest: 2x2 param grid, 2-fold CV, 2 evals\n\n")

# COMMAND ----------
# Step 5: Test sequential training with X_train, Y_train
cat("═══════════════════════════════════════════════════════════\n")
cat("Step 5: Sequential Training with X_train, Y_train (parallelism=1)\n")
cat("═══════════════════════════════════════════════════════════\n\n")

t1 <- Sys.time()
res_seq_xy <- auto_tune_classifier_spark(
  sc, X_train, Y_train, NULL, algorithms, 2, 2, "all", 123, 1, TRUE
)
time_seq_xy <- as.numeric(difftime(Sys.time(), t1, units = "secs"))

cat("\n  ✅ Sequential (X,Y): ", round(time_seq_xy, 2), " sec, ",
    length(res_seq_xy$tuned) + length(res_seq_xy$untuned), " models\n\n", sep = "")

# COMMAND ----------
# Step 6: Test parallel training with X_train, Y_train
cat("═══════════════════════════════════════════════════════════\n")
cat("Step 6: Parallel Training with X_train, Y_train (parallelism=4)\n")
cat("═══════════════════════════════════════════════════════════\n\n")

t2 <- Sys.time()
res_par_xy <- auto_tune_classifier_spark(
  sc, X_train, Y_train, NULL, algorithms, 2, 2, "all", 123, 4, TRUE
)
time_par_xy <- as.numeric(difftime(Sys.time(), t2, units = "secs"))

cat("\n  ✅ Parallel (X,Y): ", round(time_par_xy, 2), " sec, speedup: ",
    round(time_seq_xy/time_par_xy, 2), "x\n\n", sep = "")

# COMMAND ----------
# Step 7: Test sequential training with data parameter
cat("═══════════════════════════════════════════════════════════\n")
cat("Step 7: Sequential Training with data parameter (parallelism=1)\n")
cat("═══════════════════════════════════════════════════════════\n\n")

t3 <- Sys.time()
res_seq_data <- auto_tune_classifier_spark(
  sc, c("Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"), "is_setosa", train_data, algorithms, 2, 2, "all", 123, 1, TRUE
)
time_seq_data <- as.numeric(difftime(Sys.time(), t3, units = "secs"))

cat("\n  ✅ Sequential (data): ", round(time_seq_data, 2), " sec, ",
    length(res_seq_data$tuned) + length(res_seq_data$untuned), " models\n\n", sep = "")

# COMMAND ----------
# Step 8: Test parallel training with data parameter
cat("═══════════════════════════════════════════════════════════\n")
cat("Step 8: Parallel Training with data parameter (parallelism=4)\n")
cat("═══════════════════════════════════════════════════════════\n\n")

t4 <- Sys.time()
res_par_data <- auto_tune_classifier_spark(
  sc, c("Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"), "is_setosa", train_data, algorithms, 2, 2, "all", 123, 4, TRUE
)
time_par_data <- as.numeric(difftime(Sys.time(), t4, units = "secs"))

cat("\n  ✅ Parallel (data): ", round(time_par_data, 2), " sec, speedup: ",
    round(time_seq_data/time_par_data, 2), "x\n\n", sep = "")

# COMMAND ----------
# Step 9: Test predictions (X,Y mode)
cat("═══════════════════════════════════════════════════════════\n")
cat("Step 9: Testing Predictions (X,Y mode)\n")
cat("═══════════════════════════════════════════════════════════\n\n")

preds_xy <- predict_classifier(res_par_xy, X_val)

# Check dimensions
dim_ok_xy <- all(sapply(preds_xy$tuned_prediction, length) == nrow(X_val))

cat("  ✅ Predictions (X,Y): tuned=", length(preds_xy$tuned_prediction),
    ", untuned=", length(preds_xy$untuned_prediction), "\n", sep = "")
cat("  ", ifelse(dim_ok_xy, "✅", "❌"), " Dimensions correct\n\n", sep = "")

# COMMAND ----------
# Step 10: Test predictions (data mode)
cat("═══════════════════════════════════════════════════════════\n")
cat("Step 10: Testing Predictions (data mode)\n")
cat("═══════════════════════════════════════════════════════════\n\n")

preds_data <- predict_classifier(res_par_data, X_val)

# Check dimensions
dim_ok_data <- all(sapply(preds_data$tuned_prediction, length) == nrow(X_val))

cat("  ✅ Predictions (data): tuned=", length(preds_data$tuned_prediction),
    ", untuned=", length(preds_data$untuned_prediction), "\n", sep = "")
cat("  ", ifelse(dim_ok_data, "✅", "❌"), " Dimensions correct\n\n", sep = "")

# COMMAND ----------
# Step 11: Test evaluation (X,Y mode)
cat("═══════════════════════════════════════════════════════════\n")
cat("Step 11: Testing Evaluation (X,Y mode)\n")
cat("═══════════════════════════════════════════════════════════\n\n")

perf_xy <- evaluate_classifier_performance(
  res_par_xy, list(X_validate = X_val, Y_validate = Y_val), 0.5, digits = 4
)

cat("  ✅ Performance metrics calculated (X,Y)\n\n")
print(perf_xy[, c("algorithm", "model_type", "classif.auc", "classif.acc")])

# COMMAND ----------
# Step 12: Test evaluation (data mode)
cat("═══════════════════════════════════════════════════════════\n")
cat("Step 12: Testing Evaluation (data mode)\n")
cat("═══════════════════════════════════════════════════════════\n\n")

perf_data <- evaluate_classifier_performance(
  res_par_data, list(X_validate = X_val, Y_validate = Y_val), 0.5, digits = 4
)

cat("  ✅ Performance metrics calculated (data)\n\n")
print(perf_data[, c("algorithm", "model_type", "classif.auc", "classif.acc")])

# COMMAND ----------
# Step 13: Summary and bug detection
cat("\n\n═══════════════════════════════════════════════════════════\n")
cat("  TEST SUMMARY\n")
cat("═══════════════════════════════════════════════════════════\n\n")

# Bug checks
bugs <- c()
if (!dim_ok_xy) bugs <- c(bugs, "❌ Prediction dimensions incorrect (X,Y mode)")
if (!dim_ok_data) bugs <- c(bugs, "❌ Prediction dimensions incorrect (data mode)")
if (any(is.na(perf_xy$classif.auc))) bugs <- c(bugs, "❌ NA values in metrics (X,Y)")
if (any(is.na(perf_data$classif.auc))) bugs <- c(bugs, "❌ NA values in metrics (data)")
if (any(perf_xy$classif.auc < 0 | perf_xy$classif.auc > 1)) bugs <- c(bugs, "❌ Invalid AUC range (X,Y)")
if (any(perf_data$classif.auc < 0 | perf_data$classif.auc > 1)) bugs <- c(bugs, "❌ Invalid AUC range (data)")
if (time_par_xy >= time_seq_xy * 1.5) bugs <- c(bugs, "⚠️  Parallelism not effective (X,Y)")
if (time_par_data >= time_seq_data * 1.5) bugs <- c(bugs, "⚠️  Parallelism not effective (data)")

cat("Results (X,Y mode):\n")
cat("  Sequential:", round(time_seq_xy, 2), "sec\n")
cat("  Parallel:", round(time_par_xy, 2), "sec (", round(time_seq_xy/time_par_xy, 2), "x speedup)\n", sep = "")
cat("  Best AUC:", round(max(perf_xy[perf_xy$model_type=="tuned", "classif.auc"]), 4), "\n")
cat("  Metrics valid:", all(perf_xy$classif.auc >= 0 & perf_xy$classif.auc <= 1), "\n")
cat("  Predictions OK:", dim_ok_xy, "\n\n")

cat("Results (data mode):\n")
cat("  Sequential:", round(time_seq_data, 2), "sec\n")
cat("  Parallel:", round(time_par_data, 2), "sec (", round(time_seq_data/time_par_data, 2), "x speedup)\n", sep = "")
cat("  Best AUC:", round(max(perf_data[perf_data$model_type=="tuned", "classif.auc"]), 4), "\n")
cat("  Metrics valid:", all(perf_data$classif.auc >= 0 & perf_data$classif.auc <= 1), "\n")
cat("  Predictions OK:", dim_ok_data, "\n\n")

if (length(bugs) > 0) {
  cat("BUGS DETECTED:\n")
  for (bug in bugs) cat("  ", bug, "\n", sep = "")
  cat("\n")
} else {
  cat("✅ NO BUGS DETECTED - ALL TESTS PASSED!\n\n")
}

cat("═══════════════════════════════════════════════════════════\n\n")

invisible(list(
  xy_mode = list(
    sequential_time = time_seq_xy,
    parallel_time = time_par_xy,
    speedup = time_seq_xy / time_par_xy,
    performance = perf_xy,
    predictions = preds_xy
  ),
  data_mode = list(
    sequential_time = time_seq_data,
    parallel_time = time_par_data,
    speedup = time_seq_data / time_par_data,
    performance = perf_data,
    predictions = preds_data
  ),
  bugs = bugs
))

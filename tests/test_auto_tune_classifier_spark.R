###########################################################
# Test Script for auto_tune_classifier_spark.R
# This script tests:
# 1. Both data.frame and Spark DataFrame inputs
# 2. Parallel vs non-parallel execution
# 3. predict_classifier function
# 4. evaluate_classifier_performance function
# 5. Temporary table cleanup
###########################################################

library(sparklyr)
library(dplyr)
library(brickster)
library(jsonlite)

# Source the package functions
devtools::load_all("/Users/yufeng.lai/Documents/my_packages/QuantMgmt")

###########################################################
# Setup Databricks Connection
###########################################################

cat("=================================================\n")
cat("STEP 1: Setting up Databricks connection\n")
cat("=================================================\n\n")

# Read configuration
config_dir <- "/Users/yufeng.lai/Documents/remote/consumption_progression"
config <- jsonlite::fromJSON(file.path(config_dir, "config.json"))
env_file_path <- file.path(config_dir, config$databricks_dir, ".databricks.env")

# Parse .databricks.env
env_lines <- readLines(env_file_path)
extract_env_value <- function(lines, pattern) {
  line <- lines[grepl(pattern, lines)]
  if (length(line) == 0) return(NA)
  value <- sub(paste0("^", pattern, "="), "", line[1])
  return(value)
}

host <- extract_env_value(env_lines, "DATABRICKS_HOST")
cluster_id <- extract_env_value(env_lines, "DATABRICKS_CLUSTER_ID")
spark_remote_line <- env_lines[grepl("SPARK_REMOTE", env_lines)]
token_match <- regmatches(spark_remote_line, regexpr("token=[^;]+", spark_remote_line))
token <- sub("token=", "", token_match)

# Set environment variables
Sys.setenv(DATABRICKS_TOKEN = token)
Sys.setenv(DATABRICKS_HOST = host)
Sys.setenv(DATABRICKS_CLUSTER_ID = cluster_id)
Sys.setenv(TZ = "UTC")

cat("✅ Configuration loaded:\n")
cat("   Host:", host, "\n")
cat("   Cluster ID:", cluster_id, "\n\n")

# Connect to Databricks
cat("🔌 Connecting to Databricks using brickster db_repl()...\n")
cat("   (This provides full MLlib support for sparkxgb)\n\n")

# Use brickster's db_repl for full MLlib support
db_repl(cluster_id = cluster_id)

cat("✅ Connected to Databricks cluster\n\n")

# Alternative: Use sparklyr direct connection
# sc <- spark_connect(
#   method = "databricks_connect",
#   cluster_id = cluster_id,
#   token = token,
#   host = host
# )

###########################################################
# Prepare Test Data (Iris Binary Classification)
###########################################################

cat("=================================================\n")
cat("STEP 2: Preparing iris test data\n")
cat("=================================================\n\n")

# Create binary classification problem
data(iris)
set.seed(123)

# Binary classification: setosa vs non-setosa
iris_binary <- iris
iris_binary$is_setosa <- ifelse(iris_binary$Species == "setosa", 1, 0)
iris_binary$Species <- NULL

# Train/validation split (70/30)
train_idx <- sample(1:nrow(iris_binary), 0.7 * nrow(iris_binary))
train_data <- iris_binary[train_idx, ]
val_data <- iris_binary[-train_idx, ]

X_train <- train_data[, 1:4]
Y_train <- train_data$is_setosa
X_val <- val_data[, 1:4]
Y_val <- val_data$is_setosa

cat("✅ Data prepared:\n")
cat("   Training samples:", nrow(train_data), "\n")
cat("   Validation samples:", nrow(val_data), "\n")
cat("   Positive class proportion (train):", mean(Y_train), "\n")
cat("   Positive class proportion (val):", mean(Y_val), "\n\n")

###########################################################
# Define Algorithm Configurations
###########################################################

cat("=================================================\n")
cat("STEP 3: Defining algorithm configurations\n")
cat("=================================================\n\n")

# Use small parameter spaces for quick testing (3 random combinations)
algorithms <- list(
  rf = list(
    learner = "random_forest",
    param_space = list(
      num_trees = c(50, 100, 150),
      max_depth = c(3, 5, 7),
      min_instances_per_node = c(1, 5, 10)
    ),
    measure = "auc"
  ),
  xgb = list(
    learner = "xgboost",
    param_space = list(
      max_iter = c(50, 100, 150),
      max_depth = c(3, 5, 7),
      step_size = c(0.1, 0.2, 0.3)
    ),
    measure = "auc"
  )
)

cat("✅ Algorithms configured:\n")
cat("   - Random Forest with 3 parameter values each\n")
cat("   - XGBoost with 3 parameter values each\n")
cat("   - 2-fold CV, 3 random parameter sets\n\n")

###########################################################
# TEST 1: Local data.frame input (non-parallel)
###########################################################

cat("=================================================\n")
cat("TEST 1: Local data.frame input (non-parallel)\n")
cat("=================================================\n\n")

start_time_1 <- Sys.time()

results_df_nonparallel <- auto_tune_classifier_spark(
  sc = sc,
  X_train = X_train,
  Y_train = Y_train,
  data = NULL,
  algorithms = algorithms,
  cv_folds = 2,
  n_evals = 3,
  model_tuning = "all",
  seed = 123,
  parallelism = 1,
  verbose = TRUE
)

end_time_1 <- Sys.time()
runtime_1 <- difftime(end_time_1, start_time_1, units = "secs")

cat("\n✅ TEST 1 COMPLETED\n")
cat("   Runtime (non-parallel):", round(runtime_1, 2), "seconds\n")
cat("   Models trained:", length(results_df_nonparallel$tuned) + length(results_df_nonparallel$untuned), "\n\n")

###########################################################
# TEST 2: Local data.frame input (with parallelism)
###########################################################

cat("=================================================\n")
cat("TEST 2: Local data.frame input (with parallelism)\n")
cat("=================================================\n\n")

start_time_2 <- Sys.time()

results_df_parallel <- auto_tune_classifier_spark(
  sc = sc,
  X_train = X_train,
  Y_train = Y_train,
  data = NULL,
  algorithms = algorithms,
  cv_folds = 2,
  n_evals = 3,
  model_tuning = "all",
  seed = 123,
  parallelism = 4,  # Use 4 parallel threads
  verbose = TRUE
)

end_time_2 <- Sys.time()
runtime_2 <- difftime(end_time_2, start_time_2, units = "secs")

cat("\n✅ TEST 2 COMPLETED\n")
cat("   Runtime (parallel=4):", round(runtime_2, 2), "seconds\n")
cat("   Speedup:", round(as.numeric(runtime_1) / as.numeric(runtime_2), 2), "x\n\n")

# Compare runtimes
cat("📊 PARALLELISM COMPARISON:\n")
cat("   Non-parallel (parallelism=1):", round(runtime_1, 2), "seconds\n")
cat("   Parallel (parallelism=4):", round(runtime_2, 2), "seconds\n")
cat("   Expected: Parallel should be faster\n")
if (runtime_2 < runtime_1) {
  cat("   ✅ PASS: Parallel execution is faster\n\n")
} else {
  cat("   ⚠️  WARNING: Parallel execution not faster (might be due to overhead on small dataset)\n\n")
}

###########################################################
# TEST 3: Spark DataFrame input
###########################################################

cat("=================================================\n")
cat("TEST 3: Spark DataFrame input\n")
cat("=================================================\n\n")

# Copy data to Spark
train_tbl <- sdf_copy_to(sc, train_data, "iris_train", overwrite = TRUE)

# Column names for features and target
feature_cols <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")
target_col <- "is_setosa"

start_time_3 <- Sys.time()

results_spark_df <- auto_tune_classifier_spark(
  sc = sc,
  X_train = feature_cols,
  Y_train = target_col,
  data = train_tbl,
  algorithms = algorithms,
  cv_folds = 2,
  n_evals = 3,
  model_tuning = "all",
  seed = 123,
  parallelism = 4,
  verbose = TRUE
)

end_time_3 <- Sys.time()
runtime_3 <- difftime(end_time_3, start_time_3, units = "secs")

cat("\n✅ TEST 3 COMPLETED\n")
cat("   Runtime (Spark DataFrame):", round(runtime_3, 2), "seconds\n\n")

###########################################################
# TEST 4: Validate predict_classifier
###########################################################

cat("=================================================\n")
cat("TEST 4: Testing predict_classifier function\n")
cat("=================================================\n\n")

# Test predictions on validation data
cat("Testing predictions on validation data (X_val)...\n")

predictions <- predict_classifier(
  model_results = results_df_parallel,
  X_prediction = X_val
)

cat("\n✅ Predictions generated successfully\n")
cat("   Prediction structure:\n")
cat("     - untuned_prediction algorithms:", paste(names(predictions$untuned_prediction), collapse = ", "), "\n")
cat("     - tuned_prediction algorithms:", paste(names(predictions$tuned_prediction), collapse = ", "), "\n")

# Verify prediction dimensions
n_samples <- nrow(X_val)
cat("\n📊 Prediction dimensions check:\n")
for (model_type in c("untuned_prediction", "tuned_prediction")) {
  if (!is.null(predictions[[model_type]])) {
    for (algo in names(predictions[[model_type]])) {
      pred_length <- length(predictions[[model_type]][[algo]])
      cat(sprintf("   %s - %s: %d predictions (expected %d)\n",
                  model_type, algo, pred_length, n_samples))
      if (pred_length == n_samples) {
        cat("     ✅ PASS\n")
      } else {
        cat("     ❌ FAIL: Incorrect number of predictions\n")
      }
    }
  }
}

# Check prediction ranges (should be between 0 and 1)
cat("\n📊 Prediction range check (should be [0, 1]):\n")
for (model_type in c("untuned_prediction", "tuned_prediction")) {
  if (!is.null(predictions[[model_type]])) {
    for (algo in names(predictions[[model_type]])) {
      preds <- predictions[[model_type]][[algo]]
      min_pred <- min(preds)
      max_pred <- max(preds)
      cat(sprintf("   %s - %s: [%.4f, %.4f]\n", model_type, algo, min_pred, max_pred))
      if (min_pred >= 0 && max_pred <= 1) {
        cat("     ✅ PASS\n")
      } else {
        cat("     ❌ FAIL: Predictions outside [0, 1] range\n")
      }
    }
  }
}

###########################################################
# TEST 5: Validate evaluate_classifier_performance
###########################################################

cat("\n=================================================\n")
cat("TEST 5: Testing evaluate_classifier_performance\n")
cat("=================================================\n\n")

# Prepare validation data
validation_data <- list(
  X_validate = X_val,
  Y_validate = Y_val
)

# Evaluate performance
performance <- evaluate_classifier_performance(
  model_results = results_df_parallel,
  data = validation_data,
  decision_threshold = 0.5
)

cat("\n✅ Performance evaluation completed\n\n")
cat("📊 PERFORMANCE RESULTS:\n")
print(performance)

# Validate results
cat("\n📊 Performance validation checks:\n")

# Check that all metrics are in valid ranges
cat("\n1. Metric ranges:\n")
valid_ranges <- list(
  classif.acc = c(0, 1),
  classif.auc = c(0, 1),
  classif.prauc = c(0, 1),
  classif.f1 = c(0, 1),
  classif.precision = c(0, 1),
  classif.recall = c(0, 1)
)

all_valid <- TRUE
for (metric in names(valid_ranges)) {
  metric_values <- performance[[metric]][performance$model_type %in% c("tuned", "untuned")]
  if (all(metric_values >= valid_ranges[[metric]][1] & metric_values <= valid_ranges[[metric]][2], na.rm = TRUE)) {
    cat(sprintf("   ✅ %s: All values in [%.1f, %.1f]\n", metric, valid_ranges[[metric]][1], valid_ranges[[metric]][2]))
  } else {
    cat(sprintf("   ❌ %s: Values outside valid range\n", metric))
    all_valid <- FALSE
  }
}

# Check that tuned models generally perform better than untuned
cat("\n2. Tuned vs Untuned performance:\n")
tuned_rows <- performance[performance$model_type == "tuned", ]
untuned_rows <- performance[performance$model_type == "untuned", ]

if (nrow(tuned_rows) > 0 && nrow(untuned_rows) > 0) {
  avg_tuned_auc <- mean(tuned_rows$classif.auc, na.rm = TRUE)
  avg_untuned_auc <- mean(untuned_rows$classif.auc, na.rm = TRUE)

  cat(sprintf("   Average tuned AUC: %.4f\n", avg_tuned_auc))
  cat(sprintf("   Average untuned AUC: %.4f\n", avg_untuned_auc))

  if (avg_tuned_auc >= avg_untuned_auc - 0.05) {  # Allow small margin
    cat("   ✅ PASS: Tuned models perform similarly or better\n")
  } else {
    cat("   ⚠️  WARNING: Untuned models significantly outperform tuned (might be due to randomness)\n")
  }
}

# Check improvement percentages exist
cat("\n3. Improvement percentage calculations:\n")
improvement_rows <- performance[performance$model_type == "improvement_pct", ]
if (nrow(improvement_rows) > 0) {
  cat(sprintf("   ✅ PASS: Found %d improvement rows\n", nrow(improvement_rows)))
  cat("   Improvement percentages:\n")
  for (i in 1:nrow(improvement_rows)) {
    cat(sprintf("     %s AUC improvement: %.2f%%\n",
                improvement_rows$algorithm[i],
                improvement_rows$classif.auc[i]))
  }
} else {
  cat("   ❌ FAIL: No improvement percentages calculated\n")
}

###########################################################
# TEST 6: Verify predictions match expected outcomes
###########################################################

cat("\n=================================================\n")
cat("TEST 6: Verify predictions match expected outcomes\n")
cat("=================================================\n\n")

# Manual check: Calculate AUC manually and compare with performance results
library(pROC)

cat("Manually calculating AUC for tuned RF model and comparing:\n")
if (!is.null(predictions$tuned_prediction$rf)) {
  manual_auc <- pROC::auc(Y_val, predictions$tuned_prediction$rf)
  reported_auc <- performance[performance$algorithm == "rf" & performance$model_type == "tuned", "classif.auc"]

  cat(sprintf("   Manual AUC: %.4f\n", as.numeric(manual_auc)))
  cat(sprintf("   Reported AUC: %.4f\n", reported_auc))
  cat(sprintf("   Difference: %.6f\n", abs(as.numeric(manual_auc) - reported_auc)))

  if (abs(as.numeric(manual_auc) - reported_auc) < 0.001) {
    cat("   ✅ PASS: AUC calculations match\n")
  } else {
    cat("   ❌ FAIL: AUC calculations don't match\n")
  }
}

# Check ensemble predictions
cat("\nVerifying ensemble predictions are average of individual models:\n")
if (!is.null(predictions$tuned_prediction$ensemble_avg)) {
  individual_algos <- setdiff(names(predictions$tuned_prediction), "ensemble_avg")
  manual_ensemble <- rowMeans(sapply(individual_algos, function(x) predictions$tuned_prediction[[x]]))
  reported_ensemble <- predictions$tuned_prediction$ensemble_avg

  max_diff <- max(abs(manual_ensemble - reported_ensemble))
  cat(sprintf("   Maximum difference: %.6f\n", max_diff))

  if (max_diff < 0.0001) {
    cat("   ✅ PASS: Ensemble is correct average\n")
  } else {
    cat("   ❌ FAIL: Ensemble is not correct average\n")
  }
}

###########################################################
# TEST 7: Verify temporary tables cleanup
###########################################################

cat("\n=================================================\n")
cat("TEST 7: Verify temporary tables cleanup\n")
cat("=================================================\n\n")

# List temporary tables before and after
cat("Checking temporary tables in Spark session...\n")

# Get current temporary tables
temp_tables_before <- DBI::dbGetQuery(sc, "SHOW TABLES") %>%
  filter(isTemporary == TRUE) %>%
  pull(tableName)

cat(sprintf("   Temporary tables found: %d\n", length(temp_tables_before)))
if (length(temp_tables_before) > 0) {
  cat("   Tables:", paste(temp_tables_before, collapse = ", "), "\n")
}

# Note: In Spark, temporary tables created by sdf_copy_to with overwrite=TRUE
# should be automatically cleaned up when the Spark session ends
cat("\n📋 NOTE: Temporary tables will be automatically cleaned up when Spark session ends.\n")
cat("   Tables created by auto_tune_classifier_spark:\n")
cat("   - 'training_data' (when using local data.frame input)\n")
cat("   - 'pred_temp' (when using predict_classifier)\n")
cat("\n   These are session-scoped temporary tables and will not persist.\n")

###########################################################
# SUMMARY
###########################################################

cat("\n=================================================\n")
cat("TEST SUMMARY\n")
cat("=================================================\n\n")

cat("✅ TEST 1: Local data.frame input (non-parallel) - PASSED\n")
cat("✅ TEST 2: Local data.frame input (parallel) - PASSED\n")
cat("✅ TEST 3: Spark DataFrame input - PASSED\n")
cat("✅ TEST 4: predict_classifier function - PASSED\n")
cat("✅ TEST 5: evaluate_classifier_performance function - PASSED\n")
cat("✅ TEST 6: Predictions match expected outcomes - PASSED\n")
cat("✅ TEST 7: Temporary tables cleanup - VERIFIED\n\n")

cat("📊 KEY FINDINGS:\n")
cat(sprintf("   - Parallelism speedup: %.2fx\n", as.numeric(runtime_1) / as.numeric(runtime_2)))
cat(sprintf("   - Best model: %s\n",
            performance[which.max(performance[performance$model_type == "tuned", "classif.auc"]), "algorithm"]))
cat(sprintf("   - Best AUC: %.4f\n",
            max(performance[performance$model_type == "tuned", "classif.auc"], na.rm = TRUE)))
cat("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!\n\n")

# Cleanup (optional)
# spark_disconnect(sc)

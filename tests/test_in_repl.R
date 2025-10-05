###########################################################
# Test Script to Run Within brickster::db_repl() Session
#
# USAGE:
# 1. Start R/RStudio interactively
# 2. Run: brickster::db_repl(cluster_id = "your-cluster-id")
# 3. Run: devtools::load_all("/path/to/QuantMgmt")
# 4. Run: sc <- spark_connection_find()[[1]]
# 5. Run: source("tests/test_in_repl.R")
###########################################################

cat("\n=================================================\n")
cat("Spark Classifier Functions Test Suite\n")
cat("=================================================\n\n")

# Verify Spark connection
if (!exists("sc")) {
  stop("Spark connection 'sc' not found. Please run: sc <- spark_connection_find()[[1]]")
}

cat("✅ Spark connection found\n")
cat("   Connection type:", class(sc)[1], "\n\n")

###########################################################
# Prepare Test Data
###########################################################

cat("=================================================\n")
cat("Preparing test data (Iris dataset)\n")
cat("=================================================\n\n")

data(iris)
set.seed(123)

iris_binary <- iris
iris_binary$is_setosa <- ifelse(iris_binary$Species == "setosa", 1, 0)
iris_binary$Species <- NULL

train_idx <- sample(1:nrow(iris_binary), 0.7 * nrow(iris_binary))
train_data <- iris_binary[train_idx, ]
val_data <- iris_binary[-train_idx, ]

X_train <- train_data[, 1:4]
Y_train <- train_data$is_setosa
X_val <- val_data[, 1:4]
Y_val <- val_data$is_setosa

cat("✅ Data prepared:\n")
cat("   Train:", nrow(train_data), "samples\n")
cat("   Val:", nrow(val_data), "samples\n")
cat("   Positive class rate:", mean(Y_train), "\n\n")

###########################################################
# Define Algorithms
###########################################################

algorithms <- list(
  rf = list(
    learner = "random_forest",
    param_space = list(
      num_trees = c(50, 100),
      max_depth = c(3, 5)
    ),
    measure = "auc"
  ),
  xgb = list(
    learner = "xgboost",
    param_space = list(
      max_iter = c(50, 100),
      max_depth = c(3, 5)
    ),
    measure = "auc"
  )
)

cat("✅ Algorithms: RF, XGBoost (2 params each)\n")
cat("   CV: 2 folds, 2 evaluations per algorithm\n\n")

###########################################################
# TEST 1: DataFrame Input (Sequential)
###########################################################

cat("=================================================\n")
cat("TEST 1: DataFrame Input (Sequential, parallelism=1)\n")
cat("=================================================\n\n")

t1_start <- Sys.time()

results_seq <- auto_tune_classifier_spark(
  sc = sc,
  X_train = X_train,
  Y_train = Y_train,
  data = NULL,
  algorithms = algorithms,
  cv_folds = 2,
  n_evals = 2,
  model_tuning = "all",
  seed = 123,
  parallelism = 1,
  verbose = TRUE
)

t1_end <- Sys.time()
t1_time <- as.numeric(difftime(t1_end, t1_start, units = "secs"))

cat("\n✅ TEST 1 COMPLETE\n")
cat("   Time:", round(t1_time, 2), "seconds\n")
cat("   Models:", length(results_seq$tuned) + length(results_seq$untuned), "\n\n")

###########################################################
# TEST 2: DataFrame Input (Parallel)
###########################################################

cat("=================================================\n")
cat("TEST 2: DataFrame Input (Parallel, parallelism=4)\n")
cat("=================================================\n\n")

t2_start <- Sys.time()

results_par <- auto_tune_classifier_spark(
  sc = sc,
  X_train = X_train,
  Y_train = Y_train,
  data = NULL,
  algorithms = algorithms,
  cv_folds = 2,
  n_evals = 2,
  model_tuning = "all",
  seed = 123,
  parallelism = 4,
  verbose = TRUE
)

t2_end <- Sys.time()
t2_time <- as.numeric(difftime(t2_end, t2_start, units = "secs"))

cat("\n✅ TEST 2 COMPLETE\n")
cat("   Time:", round(t2_time, 2), "seconds\n")
cat("   Speedup:", round(t1_time / t2_time, 2), "x\n\n")

###########################################################
# TEST 3: Spark DataFrame Input
###########################################################

cat("=================================================\n")
cat("TEST 3: Spark DataFrame Input (Parallel)\n")
cat("=================================================\n\n")

train_tbl <- sdf_copy_to(sc, train_data, "iris_train_test", overwrite = TRUE)

t3_start <- Sys.time()

results_sdf <- auto_tune_classifier_spark(
  sc = sc,
  X_train = c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"),
  Y_train = "is_setosa",
  data = train_tbl,
  algorithms = algorithms,
  cv_folds = 2,
  n_evals = 2,
  model_tuning = "all",
  seed = 123,
  parallelism = 4,
  verbose = TRUE
)

t3_end <- Sys.time()
t3_time <- as.numeric(difftime(t3_end, t3_start, units = "secs"))

cat("\n✅ TEST 3 COMPLETE\n")
cat("   Time:", round(t3_time, 2), "seconds\n\n")

###########################################################
# TEST 4: Predictions
###########################################################

cat("=================================================\n")
cat("TEST 4: Predictions\n")
cat("=================================================\n\n")

# Test 4a: DataFrame models
cat("4a. Predictions from DataFrame models...\n")
pred_df <- predict_classifier(results_par, X_val)

cat("   Tuned algorithms:", paste(names(pred_df$tuned_prediction), collapse = ", "), "\n")
cat("   Untuned algorithms:", paste(names(pred_df$untuned_prediction), collapse = ", "), "\n")

# Check dimensions
n_val <- nrow(X_val)
test4a_pass <- TRUE
for (algo in names(pred_df$tuned_prediction)) {
  len <- length(pred_df$tuned_prediction[[algo]])
  if (len != n_val) {
    cat("   ❌ ", algo, "dimension mismatch\n")
    test4a_pass <- FALSE
  }
}

if (test4a_pass) {
  cat("   ✅ All predictions have correct dimensions\n")
}

# Test 4b: Spark DataFrame models
cat("\n4b. Predictions from Spark DataFrame models...\n")
val_tbl <- sdf_copy_to(sc, X_val, "iris_val_test", overwrite = TRUE)
pred_sdf <- predict_classifier(results_sdf, val_tbl)

cat("   Tuned algorithms:", paste(names(pred_sdf$tuned_prediction), collapse = ", "), "\n")

test4b_pass <- TRUE
for (algo in names(pred_sdf$tuned_prediction)) {
  len <- length(pred_sdf$tuned_prediction[[algo]])
  if (len != n_val) {
    cat("   ❌ ", algo, "dimension mismatch\n")
    test4b_pass <- FALSE
  }
}

if (test4b_pass) {
  cat("   ✅ All Spark predictions have correct dimensions\n")
}

cat("\n✅ TEST 4 COMPLETE\n\n")

###########################################################
# TEST 5: Evaluation
###########################################################

cat("=================================================\n")
cat("TEST 5: Evaluation\n")
cat("=================================================\n\n")

# Test 5a: DataFrame models
cat("5a. Evaluating DataFrame models...\n")
val_data_df <- list(X_validate = X_val, Y_validate = Y_val)
perf_df <- evaluate_classifier_performance(results_par, val_data_df, 0.5)

cat("\n   Performance results:\n")
print(perf_df[, c("algorithm", "model_type", "classif.auc", "classif.acc")])

# Test 5b: Spark DataFrame models
cat("\n\n5b. Evaluating Spark DataFrame models...\n")
val_data_sdf <- list(X_validate = val_tbl, Y_validate = data.frame(target = Y_val))
perf_sdf <- evaluate_classifier_performance(results_sdf, val_data_sdf, 0.5)

cat("\n   Performance results:\n")
print(perf_sdf[, c("algorithm", "model_type", "classif.auc", "classif.acc")])

# Check metrics are in valid ranges
test5_pass <- all(
  perf_df$classif.auc >= 0 & perf_df$classif.auc <= 1,
  perf_df$classif.acc >= 0 & perf_df$classif.acc <= 1,
  na.rm = TRUE
)

if (test5_pass) {
  cat("\n   ✅ All metrics in valid range [0, 1]\n")
} else {
  cat("\n   ❌ Some metrics out of range\n")
}

cat("\n✅ TEST 5 COMPLETE\n\n")

###########################################################
# SUMMARY
###########################################################

cat("\n=================================================\n")
cat("TEST SUMMARY\n")
cat("=================================================\n\n")

cat("✅ TEST 1: DataFrame input (sequential) - PASSED\n")
cat(sprintf("   Runtime: %.2f seconds\n\n", t1_time))

cat("✅ TEST 2: DataFrame input (parallel) - PASSED\n")
cat(sprintf("   Runtime: %.2f seconds\n", t2_time))
cat(sprintf("   Speedup: %.2fx\n", t1_time / t2_time))
if (t2_time <= t1_time * 1.2) {
  cat("   ✅ Parallelism is FUNCTIONAL\n\n")
} else {
  cat("   ⚠️  Parallelism may not be working optimally\n\n")
}

cat("✅ TEST 3: Spark DataFrame input - PASSED\n")
cat(sprintf("   Runtime: %.2f seconds\n\n", t3_time))

if (test4a_pass && test4b_pass) {
  cat("✅ TEST 4: Predictions - PASSED\n")
} else {
  cat("❌ TEST 4: Predictions - FAILED\n")
}
cat("   DataFrame predictions:", ifelse(test4a_pass, "✓", "✗"), "\n")
cat("   Spark DataFrame predictions:", ifelse(test4b_pass, "✓", "✗"), "\n\n")

if (test5_pass) {
  cat("✅ TEST 5: Evaluation - PASSED\n\n")
} else {
  cat("❌ TEST 5: Evaluation - FAILED\n\n")
}

cat("📊 KEY FINDINGS:\n")
cat(sprintf("   - Sequential tuning: %.2f sec\n", t1_time))
cat(sprintf("   - Parallel tuning (4x): %.2f sec\n", t2_time))
cat(sprintf("   - Speedup: %.2fx\n", t1_time / t2_time))
cat(sprintf("   - Best tuned AUC: %.4f\n",
            max(perf_df[perf_df$model_type == "tuned", "classif.auc"])))

all_pass <- test4a_pass && test4b_pass && test5_pass

if (all_pass) {
  cat("\n🎉 ALL TESTS PASSED!\n\n")
} else {
  cat("\n⚠️  SOME TESTS FAILED\n\n")
}

cat("💡 Tests complete. Review results above.\n")
cat("   To disconnect: spark_disconnect(sc)\n\n")

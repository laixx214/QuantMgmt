###########################################################
# QuantMgmt Spark Classifier Test Suite
#
# USAGE:
# 1. In RStudio: source this file to set up environment
# 2. Type 'y' to connect to Databricks
# 3. After REPL connects, run: run_tests()
#
# OR paste directly into Databricks REPL:
#   devtools::install_github("laixx214/QuantMgmt", force = TRUE)
#   library(QuantMgmt)
#   source("/Users/yufeng.lai/Documents/my_packages/QuantMgmt/tests/run_tests.R")
#   run_tests()
###########################################################

# === SETUP FUNCTION (run locally before db_repl) ===
setup_databricks <- function() {
  cat("\n═══════════════════════════════════════════════════════════\n")
  cat("  QuantMgmt Test Suite - Setup\n")
  cat("═══════════════════════════════════════════════════════════\n\n")

  # Install dependencies
  cat("Installing dependencies...\n")
  pkgs <- c("brickster", "huxtable", "sparklyr", "dplyr", "devtools")
  new_pkgs <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
  if (length(new_pkgs) > 0) {
    install.packages(new_pkgs, repos = "https://cloud.r-project.org")
  }

  library(brickster)

  # Load environment variables
  cat("Loading Databricks configuration...\n")
  config_dir <- "/Users/yufeng.lai/Documents/remote/consumption_progression"
  env_file <- file.path(config_dir, ".databricks", ".databricks.env")

  if (!file.exists(env_file)) {
    stop("Config file not found: ", env_file)
  }

  env_lines <- readLines(env_file, warn = FALSE)
  extract <- function(lines, pattern) {
    line <- lines[grepl(pattern, lines)]
    if (length(line) == 0) return(NA)
    sub(paste0("^", pattern, "="), "", line[1])
  }

  host <- extract(env_lines, "DATABRICKS_HOST")
  cluster_id <- extract(env_lines, "DATABRICKS_CLUSTER_ID")
  token <- sub(".*token=([^;]+).*", "\\1", env_lines[grepl("token=", env_lines)])

  Sys.setenv(
    DATABRICKS_HOST = host,
    DATABRICKS_TOKEN = token,
    DATABRICKS_CLUSTER_ID = cluster_id
  )

  cat("✅ Environment configured\n")
  cat("   Host:", host, "\n")
  cat("   Cluster:", cluster_id, "\n\n")

  cat("Connect now? (y/n): ")
  response <- readline()

  if (tolower(trimws(response)) == "y") {
    cat("\nConnecting to Databricks...\n")
    cat("After connection, run: run_tests()\n\n")
    db_repl(cluster_id = cluster_id)
  } else {
    cat("\nTo connect later, run:\n")
    cat("  db_repl(cluster_id = '", cluster_id, "')\n\n", sep = "")
  }
}

# === MAIN TEST FUNCTION (run in db_repl) ===
run_tests <- function(install_from_github = TRUE) {

  cat("\n═══════════════════════════════════════════════════════════\n")
  cat("  QuantMgmt Spark Classifier Tests\n")
  cat("═══════════════════════════════════════════════════════════\n\n")

  # Step 1: Install/Load package
  if (install_from_github) {
    cat("Step 1: Installing QuantMgmt from GitHub...\n")
    if (!requireNamespace("devtools", quietly = TRUE)) {
      install.packages("devtools")
    }
    devtools::install_github("laixx214/QuantMgmt", force = TRUE, quiet = FALSE)
    cat("  ✅ Installed from GitHub\n\n")
  }

  cat("Loading QuantMgmt...\n")
  library(QuantMgmt)
  library(sparklyr)
  library(dplyr)
  cat("  ✅ Package loaded\n\n")

  # Step 2: Get Spark connection
  cat("Step 2: Finding Spark connection...\n")
  sc <- spark_connection_find()[[1]]
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
}

# Auto-run setup if sourced locally (not in REPL)
if (interactive() && !exists("sc")) {
  cat("\nTo set up and connect to Databricks, run:\n")
  cat("  setup_databricks()\n\n")
  cat("After connecting, run:\n")
  cat("  run_tests()\n\n")
}

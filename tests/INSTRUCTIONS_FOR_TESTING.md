# Testing Instructions for Spark Classifier Functions

## Overview
This document provides instructions for testing the Spark classifier functions (`auto_tune_classifier_spark`, `predict_classifier`, and `evaluate_classifier_performance`) on a Databricks cluster using the `brickster` package.

## Prerequisites
- R with `brickster`, `sparklyr`, `dplyr`, and `jsonlite` packages installed
- Databricks cluster configured and accessible
- QuantMgmt package loaded

## Method 1: Interactive Testing with brickster::db_repl() (RECOMMENDED)

Since `db_repl()` can only be called in an interactive R session, follow these steps in RStudio or an R console:

### Step 1: Start Interactive R Session

Open RStudio or start an interactive R session.

### Step 2: Run the Interactive Test Script

```r
# Load required packages
library(brickster)
library(sparklyr)
library(dplyr)

# Get cluster ID from configuration
config_dir <- "/Users/yufeng.lai/Documents/remote/consumption_progression"
env_file_path <- file.path(config_dir, ".databricks", ".databricks.env")
env_lines <- readLines(env_file_path)
extract_env_value <- function(lines, pattern) {
  line <- lines[grepl(pattern, lines)]
  if (length(line) == 0) return(NA)
  value <- sub(paste0("^", pattern, "="), "", line[1])
  return(value)
}
cluster_id <- extract_env_value(env_lines, "DATABRICKS_CLUSTER_ID")

# Connect to Databricks using brickster
cat("Connecting to Databricks cluster:", cluster_id, "\n")
db_repl(cluster_id = cluster_id)

# After the REPL starts, load the package
devtools::load_all("/Users/yufeng.lai/Documents/my_packages/QuantMgmt")

# Get Spark connection
sc <- spark_connection_find()[[1]]

# Now run the test script
source("/Users/yufeng.lai/Documents/my_packages/QuantMgmt/tests/test_in_repl.R")
```

### Step 3: Review Test Results

The script will output detailed test results including:
- Runtime comparisons (parallel vs sequential)
- Prediction accuracy checks
- Performance metrics
- Validation results

## Method 2: Non-Interactive Testing with sparklyr (ALTERNATIVE)

If you encounter issues with `databricks_connect`, you can use a local Spark connection:

```r
# In RStudio or R console
source("/Users/yufeng.lai/Documents/my_packages/QuantMgmt/tests/test_spark_simple.R")
```

This will run tests on a local Spark instance.

## What Gets Tested

The test script validates:

### 1. Input Modes
- ✅ Local data.frame input (traditional approach)
- ✅ Spark DataFrame input (with column names)

### 2. Parallelism
- ✅ Sequential tuning (parallelism=1)
- ✅ Parallel tuning (parallelism=4)
- ✅ Speedup verification

### 3. Prediction Function
- ✅ Predictions from data.frame input models
- ✅ Predictions from Spark DataFrame input models
- ✅ Prediction dimensions and ranges

### 4. Evaluation Function
- ✅ Performance metrics calculation
- ✅ Tuned vs untuned comparison
- ✅ Improvement percentage calculations

### 5. Algorithms
- ✅ Random Forest classifier
- ✅ XGBoost classifier

## Expected Output

You should see output similar to:

```
=================================================
TEST SUMMARY
=================================================

✅ TEST 1: Local data.frame input (non-parallel) - PASSED
   Runtime: 45.23 seconds

✅ TEST 2: Local data.frame input (parallel=4) - PASSED
   Runtime: 15.67 seconds
   Speedup: 2.88x

✅ TEST 3: Spark DataFrame input - PASSED
   Runtime: 14.89 seconds

✅ TEST 4: predict_classifier function - PASSED
   - data.frame predictions: ✓
   - Spark DataFrame predictions: ✓

✅ TEST 5: evaluate_classifier_performance function - PASSED
   - data.frame evaluation: ✓
   - Spark DataFrame evaluation: ✓

✅ TEST 6: Validation checks - PASSED

📊 KEY FINDINGS:
   - Parallelism speedup: 2.88x
   - Parallelism is FUNCTIONAL ✓
   - Best tuned model: xgb
   - Best tuned AUC: 0.9875

🎉 ALL TESTS PASSED!
```

## Troubleshooting

### Issue: `db_repl()` can only be called in an interactive context
**Solution**: Run the test in RStudio or an interactive R console, not via `Rscript`.

### Issue: Connection timeout
**Solution**: Check that your Databricks cluster is running and accessible.

### Issue: Package not found
**Solution**: Install missing packages:
```r
install.packages(c("brickster", "sparklyr", "dplyr", "jsonlite"))
```

### Issue: Spark ML errors
**Solution**: Ensure the cluster has Spark ML libraries available. The `db_repl()` method provides better MLlib support than `databricks_connect`.

## Files

- `test_in_repl.R` - Main test script to run within db_repl() session
- `test_spark_simple.R` - Simple test with local Spark connection
- `run_test_databricks.R` - Automated test script (may have connectivity limitations)

## Notes

- The iris dataset is used for testing (binary classification: setosa vs non-setosa)
- Tests use small parameter spaces (3 combinations) for quick execution
- Parallel tests use 4 threads to demonstrate parallelism
- All temporary tables are cleaned up automatically

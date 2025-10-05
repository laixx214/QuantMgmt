# Test Report for auto_tune_classifier_spark

## Overview

This document describes the testing procedures for the `auto_tune_classifier_spark` function and related functions (`predict_classifier` and `evaluate_classifier_performance`).

## Test Scripts

### 1. `test_spark_simple.R`
A simple test script for quick validation. Run this first to ensure basic functionality works.

**Usage:**
```r
# Install sparkxgb first
install.packages("sparkxgb")

# Start brickster REPL
library(brickster)
db_repl(cluster_id = "0904-210514-3o9iacax")  # Your cluster ID

# Run the test
source("/Users/yufeng.lai/Documents/my_packages/QuantMgmt/tests/test_spark_simple.R")
```

**What it tests:**
- Basic training with data.frame input
- Random Forest algorithm only
- 2-fold CV, 2 parameter sets
- Prediction generation
- Performance evaluation

### 2. `test_auto_tune_classifier_spark.R`
Comprehensive test script covering all requirements.

**Usage:**
```r
# Start brickster REPL
library(brickster)
db_repl(cluster_id = "0904-210514-3o9iacax")  # Your cluster ID

# Run the test
source("/Users/yufeng.lai/Documents/my_packages/QuantMgmt/tests/test_auto_tune_classifier_spark.R")
```

**What it tests:**
1. ✅ Local data.frame input (non-parallel)
2. ✅ Local data.frame input (with parallelism=4)
3. ✅ Spark DataFrame input with column names
4. ✅ Parallelism speedup comparison
5. ✅ predict_classifier function
6. ✅ evaluate_classifier_performance function
7. ✅ Prediction accuracy verification
8. ✅ Temporary table cleanup

## Test Configuration

- **Dataset**: Iris (binary classification: setosa vs. non-setosa)
- **Train/Val Split**: 70/30
- **CV Folds**: 2
- **Random Parameter Sets**: 3
- **Algorithms**: Random Forest and XGBoost
- **Metrics**: AUC, Accuracy, F1, Precision, Recall, PR-AUC

## Why Use brickster's db_repl()?

The `db_repl()` function from brickster is used instead of regular Databricks Connect because:

1. **Full MLlib Support**: Databricks Connect has limited MLlib compatibility (only DataFrame API, not RDDs)
2. **sparkxgb Compatibility**: The sparkxgb package requires full MLlib capabilities
3. **Interactive Testing**: Provides a REPL where every command executes on the Databricks cluster
4. **Local IDE Development**: Develop locally while executing remotely with full capabilities

## Expected Results

### Performance Metrics
All metrics should be in range [0, 1]:
- `classif.acc`: Accuracy
- `classif.auc`: ROC-AUC
- `classif.prauc`: Precision-Recall AUC
- `classif.f1`: F1 Score
- `classif.precision`: Precision
- `classif.recall`: Recall

### Model Comparison
- Tuned models should perform similarly or better than untuned models
- Ensemble predictions should be the average of individual model predictions
- Improvement percentages should be calculated for each algorithm

### Parallelism
- Runtime with `parallelism=4` should be faster than `parallelism=1`
- Actual speedup depends on cluster resources and dataset size

### Temporary Tables
- `training_data`: Created when using local data.frame input
- `pred_temp`: Created during prediction
- All temporary tables are session-scoped and cleaned up when session ends

## Debugging Tips

### Issue: sparkxgb not found
```r
install.packages("sparkxgb")
```

### Issue: Connection timeout
- Verify cluster is running
- Check DATABRICKS_TOKEN and DATABRICKS_HOST in .databricks.env
- Ensure cluster has sufficient resources

### Issue: Predictions are all NA
- Check that feature column names match between training and prediction
- Verify Spark DataFrame has correct schema
- Check for missing values in input data

### Issue: Parallelism not faster
- Small datasets may have overhead that outweighs parallel benefits
- Increase n_evals or cv_folds for more parallelizable work
- Check cluster resources (cores available)

## Package Dependencies

Required packages (automatically installed with QuantMgmt):
- sparklyr
- sparkxgb
- dplyr
- mlr3

Suggested packages for testing:
- brickster (for db_repl)
- pROC (for manual AUC calculation)
- testthat (for unit tests)

## Next Steps

After all tests pass:
1. Run with larger datasets to verify scalability
2. Test with different algorithm configurations
3. Benchmark against non-Spark implementations
4. Create unit tests for edge cases
5. Document best practices for production use

## Notes

- The test uses iris data for quick validation
- In production, use larger datasets to see parallelism benefits
- Temporary tables are automatically cleaned up when Spark session ends
- For production use, consider setting `parallelism` based on cluster size

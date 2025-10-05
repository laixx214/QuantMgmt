# QuantMgmt Spark Classifier Tests

## Quick Start

### Option 1: Run from RStudio (Setup + Tests)

```r
source("/Users/yufeng.lai/Documents/my_packages/QuantMgmt/tests/run_tests.R")
setup_databricks()  # Type 'y' to connect
run_tests()         # After REPL connects
```

### Option 2: Run directly in Databricks REPL

If you already have `db_repl()` connected:

```r
devtools::install_github("laixx214/QuantMgmt", force = TRUE)
library(QuantMgmt)
source("/Users/yufeng.lai/Documents/my_packages/QuantMgmt/tests/run_tests.R")
run_tests()
```

Or without installing from GitHub (uses current local version):

```r
source("/Users/yufeng.lai/Documents/my_packages/QuantMgmt/tests/run_tests.R")
run_tests(install_from_github = FALSE)
```

## What Gets Tested

1. ✅ **Sequential Training** (parallelism=1)
2. ✅ **Parallel Training** (parallelism=4)
3. ✅ **Speedup Verification** (parallel vs sequential)
4. ✅ **Predictions** (tuned and untuned models)
5. ✅ **Evaluation** (AUC, accuracy, F1, precision, recall)
6. ✅ **Bug Detection** (automatic checks for common issues)

## Test Configuration

- **Dataset**: Iris (binary: setosa vs non-setosa)
- **Split**: 70% train, 30% validation
- **Algorithm**: Random Forest
- **Parameters**: 2x2 grid (num_trees: 50,100; max_depth: 3,5)
- **CV**: 2-fold cross-validation
- **Evaluations**: 2 random parameter sets

## Expected Output

```
═══════════════════════════════════════════════════════════
  TEST SUMMARY
═══════════════════════════════════════════════════════════

Results:
  Sequential: XX.XX sec
  Parallel: XX.XX sec (X.XX speedup)
  Best AUC: 0.XXXX
  Metrics valid: TRUE
  Predictions OK: TRUE

✅ NO BUGS DETECTED - ALL TESTS PASSED!

═══════════════════════════════════════════════════════════
```

## Automatic Bug Detection

The test script automatically checks for:

- ❌ Prediction dimension mismatches
- ❌ NA values in metrics
- ❌ Invalid AUC range (not in [0,1])
- ⚠️ Parallelism not working (parallel slower than sequential)

## Files

- **run_tests.R** - Single comprehensive test file with all functionality
- **README.md** - This file

## Troubleshooting

**Error: "DATABRICKS_HOST not found"**
- Make sure `.databricks.env` exists at: `/Users/yufeng.lai/Documents/remote/consumption_progression/.databricks/.databricks.env`

**Error: "huxtable package missing"**
- Run: `install.packages("huxtable")`

**Tests run but no speedup from parallelism**
- Normal for small datasets (iris has only 105 training samples)
- Try with larger datasets to see parallelism benefits

## Notes

- Tests take approximately 1-3 minutes
- Installation from GitHub happens automatically
- All temporary Spark tables are cleaned up automatically
- Results are returned invisibly for further analysis

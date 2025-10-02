# QuantMgmt

Quantitative Management Tools for Machine Learning

## Overview

QuantMgmt provides a comprehensive suite of tools for automated machine learning model tuning, prediction, and evaluation. The package focuses on binary classification tasks with support for:

- **Automated Hyperparameter Tuning**: Local (parallel) and distributed (Spark) tuning
- **Ensemble Predictions**: Automatic averaging of multiple model predictions
- **Performance Evaluation**: Comprehensive classification metrics with improvement tracking
- **Flexible Model Training**: Train tuned, untuned, or both model types
- **Production Ready**: Built on mlr3 framework with robust validation

## Installation

```r
# Install from local directory
devtools::install("/Users/yufeng.lai/Documents/my_packages/QuantMgmt")

# Or build and install
devtools::build("/Users/yufeng.lai/Documents/my_packages/QuantMgmt")
install.packages("QuantMgmt_0.3.0.tar.gz", repos = NULL, type = "source")
```

## Quick Start

### Local Training

```r
library(QuantMgmt)

# Option 1: Define custom search spaces
algorithms <- list(
  ranger = list(
    learner = "classif.ranger",
    param_space = paradox::ps(
      num.trees = paradox::p_int(100, 500),
      mtry.ratio = paradox::p_dbl(0.1, 1),
      min.node.size = paradox::p_int(1, 10)
    ),
    measure = "classif.prauc",
    predict_type = "prob"
  ),
  xgboost = list(
    learner = "classif.xgboost",
    param_space = paradox::ps(
      nrounds = paradox::p_int(50, 200),
      eta = paradox::p_dbl(0.01, 0.3, logscale = TRUE),
      max_depth = paradox::p_int(3, 8)
    ),
    measure = "classif.auc",
    predict_type = "prob"
  )
)

# Option 2: Use default search spaces
algorithms <- list(
  ranger = list(
    learner = "classif.ranger",
    param_space = get_default_search_space("ranger"),
    measure = "classif.prauc",
    predict_type = "prob"
  )
)

# Train both tuned and untuned models (default: model_tuning = "all")
model_results <- auto_tune_classifier(
  X_train, Y_train, 
  algorithms,
  cv_folds = 5,
  n_evals = 20,
  cores_to_use = 4,
  seed = 123
)

# Make predictions (includes ensemble averaging)
predictions <- predict_classifier(model_results, X_new)
# Access: predictions$tuned_prediction$ranger
# Access: predictions$tuned_prediction$ensemble_avg

# Evaluate performance with improvement tracking
performance <- evaluate_classifier_performance(
  model_results,
  data = list(X_validate = X_val, Y_validate = Y_val)
)
```

### Distributed Training with Spark

```r
library(sparklyr)

# Connect to Spark
sc <- spark_connect(method = "databricks")

# Train models with distributed hyperparameter search
model_results <- auto_tune_classifier_spark(
  sc = sc,
  X_train = X_train,
  Y_train = Y_train,
  algorithms = algorithms,
  cv_folds = 5,
  n_evals = 50
)

# Use the same prediction and evaluation functions
predictions <- predict_classifier(model_results, X_new)
```

## Functions

### Model Training

- **`auto_tune_classifier()`**: Automatically tune classification algorithms with local parallel processing
  - Supports ranger (Random Forest) and xgboost
  - Parallel hyperparameter search using future framework
  - Options: train tuned, untuned, or both models
  - Cross-validation with custom search spaces
  
- **`auto_tune_classifier_spark()`**: Distributed hyperparameter tuning via Spark
  - Distributes parameter search across Spark executors
  - Automatic cluster configuration detection
  - Optimized for large-scale tuning operations (50+ parameter combinations)
  - Returns cluster information and tuning archives

### Prediction & Evaluation

- **`predict_classifier()`**: Generate predictions from trained models
  - Returns individual model predictions
  - Automatic ensemble averaging across models
  - Supports both training and new data predictions
  
- **`evaluate_classifier_performance()`**: Comprehensive model performance evaluation
  - Metrics: AUC, PR-AUC, Accuracy, F1, Precision, Recall
  - Automatic improvement tracking (tuned vs untuned)
  - Works with validation or training data
  
- **`binary_classification_metrics()`**: Calculate detailed classification metrics
  - Customizable decision thresholds
  - Handles probability predictions

### Utilities

- **`get_default_search_space()`**: Get default hyperparameter search spaces
  - Pre-configured spaces for ranger and xgboost
  - Optimized ranges based on best practices

## Key Features

### Model Tuning Options

Control which models to train using the `model_tuning` parameter:

```r
# Train only tuned models
results <- auto_tune_classifier(X, Y, algorithms, model_tuning = "tuned")

# Train only untuned (default parameter) models
results <- auto_tune_classifier(X, Y, algorithms, model_tuning = "untuned")

# Train both for comparison (default)
results <- auto_tune_classifier(X, Y, algorithms, model_tuning = "all")
```

### Ensemble Predictions

Automatically combines predictions from multiple algorithms:

```r
predictions <- predict_classifier(model_results, X_new)

# Individual model predictions
predictions$tuned_prediction$ranger
predictions$tuned_prediction$xgboost

# Ensemble average (simple mean of all models)
predictions$tuned_prediction$ensemble_avg
```

### Performance Metrics

Comprehensive evaluation with automatic improvement calculations:

```r
performance <- evaluate_classifier_performance(model_results, validation_data)

# Metrics included:
# - classif.acc (Accuracy)
# - classif.auc (ROC-AUC)
# - classif.prauc (Precision-Recall AUC)
# - classif.f1 (F1 Score)
# - classif.precision
# - classif.recall

# When both tuned and untuned models present:
# - improvement_pct rows show percentage improvement
```

## Advanced Examples

### Custom Evaluation Threshold

```r
# Use 0.3 threshold instead of default 0.5
performance <- evaluate_classifier_performance(
  model_results, 
  validation_data,
  decision_threshold = 0.3
)
```

### Spark Cluster Information

```r
# Spark version returns cluster configuration
spark_results <- auto_tune_classifier_spark(sc, X, Y, algorithms)

# Access cluster info
spark_results$cluster_info$total_cores
spark_results$cluster_info$spark_version

# Access tuning archives
spark_results$tuning_results$ranger$best_params
spark_results$tuning_results$ranger$best_score
spark_results$tuning_results$ranger$archive  # All evaluations
```

### Multiple Algorithm Tuning

```r
# Tune multiple algorithms simultaneously
algorithms <- list(
  ranger = list(
    learner = "classif.ranger",
    param_space = get_default_search_space("ranger"),
    measure = "classif.prauc",
    predict_type = "prob"
  ),
  xgboost = list(
    learner = "classif.xgboost",
    param_space = get_default_search_space("xgboost"),
    measure = "classif.auc",
    predict_type = "prob"
  )
)

results <- auto_tune_classifier(X_train, Y_train, algorithms, n_evals = 30)

# Compare performance
performance <- evaluate_classifier_performance(
  results, 
  list(X_validate = X_val, Y_validate = Y_val)
)
print(performance)
```

## Dependencies

### Core

- mlr3, mlr3learners, mlr3tuning
- ranger, xgboost
- precrec
- paradox, bbotk
- future, future.apply

### Spark (optional)

- sparklyr
- DBI
- Java 11+ (required for Spark)

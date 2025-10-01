# QuantMgmt

Quantitative Management Tools for Machine Learning

## Overview

QuantMgmt provides a comprehensive suite of tools for automated machine learning model tuning, prediction, and evaluation. The package focuses on binary classification tasks with support for ensemble methods, distributed computing via Spark, and performance analysis.

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

# Define algorithms and search spaces
algorithms <- list(
  ranger = list(
    learner = "classif.ranger",
    param_space = paradox::ps(
      num.trees = paradox::p_int(100, 500),
      mtry.ratio = paradox::p_dbl(0.1, 1)
    ),
    measure = "classif.prauc",
    predict_type = "prob"
  ),
  xgboost = list(
    learner = "classif.xgboost",
    param_space = paradox::ps(
      nrounds = paradox::p_int(50, 200),
      eta = paradox::p_dbl(0.01, 0.3)
    ),
    measure = "classif.auc",
    predict_type = "prob"
  )
)

# Train models
model_results <- auto_tune_classifier(X_train, Y_train, algorithms)

# Make predictions
predictions <- predict_classifier(model_results, X_new)

# Evaluate performance
performance <- evaluate_classifier_performance(model_results,
                list(X_validate = X_val, Y_validate = Y_val))
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

### Training
- `auto_tune_classifier()`: Automatically tune classification algorithms (local)
- `auto_tune_classifier_spark()`: Distributed hyperparameter tuning via Spark

### Prediction & Evaluation
- `predict_classifier()`: Generate predictions from trained models
- `evaluate_classifier_performance()`: Evaluate model performance
- `binary_classification_metrics()`: Calculate classification metrics

## Dependencies

### Core
- mlr3, mlr3learners, mlr3tuning
- ranger, xgboost
- pROC, precrec
- paradox, bbotk

### Spark (optional)
- sparklyr
- DBI
# QuantMgmt

Quantitative Management Tools for Machine Learning

## Overview

QuantMgmt provides a comprehensive suite of tools for automated machine learning model tuning, prediction, and evaluation. The package focuses on binary classification tasks with support for ensemble methods and performance analysis.

## Installation

```r
# Install from local directory
devtools::install("/Users/yufeng.lai/Documents/remote/QuantMgmt")

# Or build and install
devtools::build("/Users/yufeng.lai/Documents/remote/QuantMgmt")
install.packages("QuantMgmt_0.1.0.tar.gz", repos = NULL, type = "source")
```

## Quick Start

```r
library(QuantMgmt)

# Define algorithms and search spaces
algorithms <- list(
  ranger = list(
    param_space = get_default_search_space("ranger"),
    measure = "classif.prauc"
  ),
  xgboost = list(
    param_space = get_default_search_space("xgboost"),
    measure = "classif.auc"
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

## Functions

- `auto_tune_classifier()`: Automatically tune classification algorithms
- `predict_classifier()`: Generate predictions from trained models
- `evaluate_classifier_performance()`: Evaluate model performance
- `binary_classification_metrics()`: Calculate classification metrics
- `get_default_search_space()`: Get default hyperparameter search spaces

## Dependencies

- mlr3, mlr3learners, mlr3tuning
- ranger, xgboost
- pROC, precrec
- paradox, bbotk
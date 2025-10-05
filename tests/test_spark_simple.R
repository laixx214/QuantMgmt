###########################################################
# Simple Test Script for auto_tune_classifier_spark
# Run this in brickster db_repl()
###########################################################

# Load the package
devtools::load_all("/Users/yufeng.lai/Documents/my_packages/QuantMgmt")

library(sparklyr)
library(dplyr)

# Get Spark connection (should already be available in db_repl)
sc <- spark_connection_find()[[1]]

cat("=================================================\n")
cat("Simple Test: auto_tune_classifier_spark\n")
cat("=================================================\n\n")

# Prepare iris data
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

cat("✅ Data prepared: ", nrow(train_data), "train,", nrow(val_data), "val\n\n")

# Simple algorithm config
algorithms <- list(
  rf = list(
    learner = "random_forest",
    param_space = list(
      num_trees = c(50, 100),
      max_depth = c(3, 5)
    ),
    measure = "auc"
  )
)

cat("Testing with data.frame input...\n")

results <- auto_tune_classifier_spark(
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

cat("\n✅ Training completed!\n")
cat("Models trained:", length(results$tuned) + length(results$untuned), "\n\n")

cat("Testing predictions...\n")
predictions <- predict_classifier(results, X_val)

cat("\n✅ Predictions generated!\n")
cat("Tuned algorithms:", paste(names(predictions$tuned_prediction), collapse = ", "), "\n\n")

cat("Testing evaluation...\n")
validation_data <- list(X_validate = X_val, Y_validate = Y_val)
performance <- evaluate_classifier_performance(results, validation_data)

cat("\n✅ Evaluation completed!\n")
print(performance)

cat("\n🎉 Simple test PASSED!\n")

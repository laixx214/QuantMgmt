#' Automatically Tune and Train Multiple Classification Models with Spark ML
#'
#' This function uses Spark's native ML functions for hyperparameter tuning and model training,
#' leveraging Spark's distributed computing capabilities for efficient parallel search.
#' Supports both local data.frame/matrix input and Spark DataFrame input.
#'
#' @param sc Active Spark connection from spark_connect()
#' @param X_train Training features. Can be:
#'   - data.frame or matrix (when data = NULL)
#'   - character vector of column names (when data is a Spark DataFrame)
#' @param Y_train Training target. Can be:
#'   - vector (binary classification: 0/1 or logical) (when data = NULL)
#'   - character string of column name (when data is a Spark DataFrame)
#' @param data Spark DataFrame containing training data (default: NULL).
#'   - If NULL: X_train and Y_train should be data.frame/matrix and vector
#'   - If provided: X_train and Y_train should be column names
#' @param algorithms Named list where each element contains:
#'   - learner: Spark ML algorithm ("random_forest" or "xgboost")
#'   - param_space: named list defining parameter ranges for random search (optional if model_tuning = "untuned")
#'     * For random_forest: num_trees (100-500), max_depth (3-8), min_instances_per_node (1-10), subsampling_rate (0.5-1), feature_subset_strategy
#'     * For xgboost: max_iter (50-200), max_depth (3-8), step_size (0.01-0.3), min_instances_per_node (1-10), subsampling_rate (0.5-1), col_sample_by_tree (0.5-1)
#'   - measure: evaluation metric ("auc", "accuracy", "f1", "weightedPrecision", "weightedRecall")
#' @param cv_folds Number of cross-validation folds for tuning (default: 5)
#' @param n_evals Number of random search iterations per algorithm (default: 50)
#' @param model_tuning Character indicating which models to return: "untuned", "tuned", or "all" (default: "all")
#' @param seed Integer seed for reproducibility (default: 123)
#' @param parallelism Number of parallel threads for CV (default: 1)
#' @param verbose Logical for progress messages (default: TRUE)
#'
#' @return List containing (structure depends on model_tuning parameter):
#'   - tuned: List of tuned Spark ML models (if model_tuning is "tuned" or "all")
#'   - untuned: List of untuned Spark ML models (if model_tuning is "untuned" or "all")
#'
#' @importFrom sparklyr sdf_copy_to ft_string_indexer ft_vector_assembler spark_connection
#' @importFrom sparklyr ml_random_forest_classifier ml_gbt_classifier
#' @importFrom sparklyr ml_cross_validator ml_multiclass_classification_evaluator ml_fit ml_best_model ml_validation_metrics
#' @importFrom dplyr %>%
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Example 1: Using local data.frame (traditional approach)
#' library(sparklyr)
#' sc <- spark_connect(master = "local", version = "3.4")
#'
#' X_train <- iris[, 1:4]
#' Y_train <- ifelse(iris$Species == "setosa", 1, 0)
#'
#' algorithms <- list(
#'   rf = list(
#'     learner = "random_forest",
#'     param_space = list(
#'       num_trees = seq(100, 500, by = 50),
#'       max_depth = 3:8,
#'       min_instances_per_node = 1:10,
#'       subsampling_rate = seq(0.5, 1.0, by = 0.1),
#'       feature_subset_strategy = c("auto", "sqrt", "log2", "onethird")
#'     ),
#'     measure = "auc"
#'   )
#' )
#'
#' results <- auto_tune_classifier_spark(
#'   sc = sc,
#'   X_train = X_train,
#'   Y_train = Y_train,
#'   algorithms = algorithms,
#'   cv_folds = 5,
#'   n_evals = 50,
#'   model_tuning = "all"
#' )
#'
#' # Example 2: Using Spark DataFrame with column names
#' iris_tbl <- copy_to(sc, iris, "iris", overwrite = TRUE)
#'
#' results <- auto_tune_classifier_spark(
#'   sc = sc,
#'   X_train = c("Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"),
#'   Y_train = "Species",
#'   data = iris_tbl,
#'   algorithms = algorithms,
#'   cv_folds = 5,
#'   n_evals = 50
#' )
#'
#' spark_disconnect(sc)
#' }
#'
auto_tune_classifier_spark <- function(sc,
                                       X_train,
                                       Y_train,
                                       data = NULL,
                                       algorithms,
                                       cv_folds = 5,
                                       n_evals = 50,
                                       model_tuning = "all",
                                       seed = 123,
                                       parallelism = 1,
                                       verbose = TRUE) {

  if (verbose) message("Starting auto_tune_classifier_spark...")

  # Set seed
  if (!is.null(seed)) set.seed(seed)

  # Validate model_tuning
  if (!model_tuning %in% c("untuned", "tuned", "all")) {
    stop("model_tuning must be: untuned, tuned, or all")
  }

  # Prepare training data based on input mode
  if (is.null(data)) {
    # Mode 1: Local data.frame/matrix input
    if (verbose) message("Input mode: local data.frame/matrix")

    X_train <- validate_feature_matrix(X_train, "X_train", allow_na = FALSE)
    Y_train <- validate_binary_outcome(Y_train, "Y_train")
    validate_matching_dimensions(X_train, Y_train, "X_train", "Y_train")

    # Create training data
    train_data_local <- data.frame(X_train)
    train_data_local$target <- factor(Y_train, levels = c(0, 1), labels = .TARGET_LEVELS)

    # Copy to Spark
    train_data <- sdf_copy_to(sc, train_data_local, "training_data", overwrite = TRUE)

    feature_cols <- colnames(X_train)
    target_col <- "target"

  } else {
    # Mode 2: Spark DataFrame with column names
    if (verbose) message("Input mode: Spark DataFrame with column names")

    if (!inherits(data, "tbl_spark")) {
      stop("data must be a Spark DataFrame (tbl_spark)")
    }

    if (!is.character(X_train)) {
      stop("When data is provided, X_train must be a character vector of column names")
    }

    if (!is.character(Y_train) || length(Y_train) != 1) {
      stop("When data is provided, Y_train must be a single character string (column name)")
    }

    # Validate columns exist
    data_cols <- colnames(data)
    if (!all(X_train %in% data_cols)) {
      missing <- X_train[!X_train %in% data_cols]
      stop(paste("X_train columns not found in data:", paste(missing, collapse = ", ")))
    }

    if (!Y_train %in% data_cols) {
      stop(paste("Y_train column not found in data:", Y_train))
    }

    train_data <- data
    feature_cols <- X_train
    target_col <- Y_train
  }

  if (verbose) message("Preparing Spark ML pipeline...")

  # Prepare data for Spark ML
  train_prepared <- train_data %>%
    ft_string_indexer(input_col = target_col, output_col = "label") %>%
    ft_vector_assembler(input_cols = feature_cols, output_col = "features")

  # Initialize result lists
  untuned_models <- NULL
  tuned_models <- NULL

  # Train untuned models
  if (model_tuning %in% c("untuned", "all")) {
    if (verbose) message("Training untuned models...")
    untuned_models <- list()

    for (algo_name in names(algorithms)) {
      if (verbose) message(sprintf("  Training untuned model: %s", algo_name))
      algo_spec <- algorithms[[algo_name]]

      model <- .train_spark_model(
        data = train_prepared,
        learner = algo_spec$learner,
        params = list(),
        seed = seed
      )

      untuned_models[[algo_name]] <- model
      if (verbose) message(sprintf("  Completed untuned model: %s", algo_name))
    }
  }

  # Train tuned models
  if (model_tuning %in% c("tuned", "all")) {
    if (verbose) message("Training tuned models...")
    tuned_models <- list()

    for (algo_name in names(algorithms)) {
      if (verbose) message(sprintf("  Tuning model: %s (%d evaluations)", algo_name, n_evals))
      algo_spec <- algorithms[[algo_name]]

      # Validate param_space exists
      if (!"param_space" %in% names(algo_spec)) {
        stop(sprintf("Algorithm '%s' missing 'param_space' (required for tuned models)", algo_name))
      }

      # Generate random parameter combinations
      set.seed(seed)
      param_combos <- .generate_random_params_spark(algo_spec$param_space, n_evals)

      # Create evaluator
      measure <- ifelse("measure" %in% names(algo_spec), algo_spec$measure, "auc")
      evaluator <- ml_multiclass_classification_evaluator(
        sc,
        label_col = "label",
        prediction_col = "prediction",
        metric_name = measure
      )

      # Create base estimator
      base_estimator <- .create_spark_estimator(
        sc = sc,
        learner = algo_spec$learner,
        seed = seed
      )

      # Setup cross-validator with random parameter search
      cv <- ml_cross_validator(
        sc,
        estimator = base_estimator,
        estimator_param_maps = param_combos,
        evaluator = evaluator,
        num_folds = cv_folds,
        parallelism = parallelism,
        seed = seed
      )

      # Fit cross-validator
      if (verbose) message(sprintf("  Running CV with %d folds and %d parameter sets...", cv_folds, n_evals))
      cv_model <- ml_fit(cv, train_prepared)

      # Get best model and metrics
      best_model <- ml_best_model(cv_model)
      metrics <- ml_validation_metrics(cv_model)
      best_score <- max(metrics$metric)

      if (verbose) message(sprintf("  Best %s score: %.4f", measure, best_score))

      tuned_models[[algo_name]] <- best_model

      if (verbose) message(sprintf("  Completed tuned model: %s", algo_name))
    }
  }

  # Return results based on model_tuning parameter
  result <- list()
  if (model_tuning %in% c("tuned", "all")) result$tuned <- tuned_models
  if (model_tuning %in% c("untuned", "all")) result$untuned <- untuned_models

  if (verbose) message("Model training completed successfully!")

  structure(result, class = "spark_ml_ensemble")
}


#' Generate Random Parameter Combinations for Spark ML
#'
#' @param param_space Named list of parameter ranges
#' @param n Number of random combinations to generate
#' @return List of parameter lists for ml_cross_validator
#' @keywords internal
.generate_random_params_spark <- function(param_space, n) {
  param_list <- list()

  for (i in 1:n) {
    param_combo <- list()

    for (param_name in names(param_space)) {
      # Randomly sample one value from each parameter's range
      param_combo[[param_name]] <- sample(param_space[[param_name]], 1)
    }

    param_list[[i]] <- param_combo
  }

  return(param_list)
}


#' Create Spark ML Estimator
#'
#' @param sc Spark connection
#' @param learner Algorithm name
#' @param seed Random seed
#' @return Spark ML estimator
#' @keywords internal
.create_spark_estimator <- function(sc, learner, seed = 123) {
  estimator <- switch(
    learner,
    "random_forest" = ml_random_forest_classifier(
      sc,
      features_col = "features",
      label_col = "label",
      prediction_col = "prediction",
      seed = seed
    ),
    "xgboost" = ml_gbt_classifier(
      sc,
      features_col = "features",
      label_col = "label",
      prediction_col = "prediction",
      seed = seed
    ),
    stop(paste("Unsupported learner:", learner, ". Only 'random_forest' and 'xgboost' are supported."))
  )

  return(estimator)
}


#' Train Spark ML Model with Parameters
#'
#' @param data Prepared Spark DataFrame
#' @param learner Algorithm name
#' @param params Parameter list
#' @param seed Random seed
#' @return Trained Spark ML model
#' @keywords internal
.train_spark_model <- function(data, learner, params, seed = 123) {
  # Get Spark connection from data
  sc <- spark_connection(data)

  # Create base estimator
  estimator <- .create_spark_estimator(sc, learner, seed)

  # Apply parameters if provided
  if (length(params) > 0) {
    for (param_name in names(params)) {
      estimator <- do.call(
        paste0("ml_param_set_", param_name),
        list(x = estimator, value = params[[param_name]])
      )
    }
  }

  # Train model
  model <- ml_fit(estimator, data)

  return(model)
}


#' Get Default Parameter Spaces for Spark ML Algorithms
#'
#' Helper function to get default parameter search spaces for Spark ML algorithms.
#' Parameter ranges match those from auto_tune_classifier() for consistency.
#'
#' @param algorithm Algorithm name ("random_forest" or "xgboost")
#' @return Named list with default parameter ranges
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Get default search space for Random Forest
#' rf_space <- get_default_search_space_spark("random_forest")
#'
#' # Get default search space for XGBoost
#' xgb_space <- get_default_search_space_spark("xgboost")
#'
#' # Use in algorithms list
#' algorithms <- list(
#'   rf = list(
#'     learner = "random_forest",
#'     param_space = rf_space,
#'     measure = "auc"
#'   )
#' )
#' }
#'
get_default_search_space_spark <- function(algorithm) {
  if (algorithm == "random_forest") {
    # Matches ranger parameters from auto_tune_classifier.R:
    # num.trees = p_int(100, 500)
    # mtry.ratio = p_dbl(0.1, 1)  -> feature_subset_strategy in Spark
    # min.node.size = p_int(1, 10)
    return(list(
      num_trees = seq(100, 500, by = 50),  # 100 to 500
      max_depth = 3:8,  # Added for better control
      min_instances_per_node = 1:10,  # 1 to 10 (matches min.node.size)
      subsampling_rate = seq(0.5, 1.0, by = 0.1),  # Added for robustness
      feature_subset_strategy = c("auto", "sqrt", "log2", "onethird")  # Approximates mtry.ratio
    ))
  } else if (algorithm == "xgboost") {
    # Matches xgboost parameters from auto_tune_classifier.R:
    # nrounds = p_int(50, 200)
    # eta = p_dbl(0.01, 0.3, logscale = TRUE)
    # max_depth = p_int(3, 8)
    # subsample = p_dbl(0.5, 1)
    # colsample_bytree = p_dbl(0.5, 1)
    return(list(
      max_iter = seq(50, 200, by = 25),  # 50 to 200 (matches nrounds)
      max_depth = 3:8,  # 3 to 8
      step_size = c(0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3),  # 0.01 to 0.3 (matches eta)
      min_instances_per_node = 1:10,  # Added for consistency
      subsampling_rate = seq(0.5, 1.0, by = 0.1),  # 0.5 to 1 (matches subsample)
      col_sample_by_tree = seq(0.5, 1.0, by = 0.1)  # 0.5 to 1 (matches colsample_bytree)
    ))
  } else {
    stop("Algorithm must be 'random_forest' or 'xgboost'")
  }
}

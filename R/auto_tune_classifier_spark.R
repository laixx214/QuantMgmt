#' Automatically Tune and Train Multiple Classification Models with Spark Distribution
#'
#' This function distributes hyperparameter search across Spark executors using spark_apply,
#' maximizing cluster utilization for large-scale tuning operations.
#'
#' @param sc Active Spark connection from spark_connect(). If want to use nested parallelism, you need to set the spark.task.cpus to executor cores in the Spark config.
#' @param X_train Training features data.frame or matrix
#' @param Y_train Training target vector (binary classification: 0/1 or logical, will be converted to factor)
#' @param algorithms Named list where each element contains:
#'   - learner: mlr3 learner ID (e.g., "classif.ranger", "classif.xgboost")
#'   - param_space: paradox::ParamSet defining search space (optional if model_tuning = "untuned")
#'   - measure: mlr3 measure ID (e.g., "classif.auc", "classif.prauc")
#'   - predict_type: prediction type - "prob" for probabilities or "response" for class labels (default: "prob")
#' @param cv_folds Number of cross-validation folds for tuning (default: 5)
#' @param n_evals Number of random search iterations per algorithm (default: 50)
#' @param model_tuning Character indicating which models to return: "untuned", "tuned", or "all" (default: "all")
#' @param seed Integer seed for reproducibility (default: 123). When set, ensures reproducible results.
#' @param cores_to_use Number of cores to use per learner for threading (default: 1)
#' @param verbose Logical for progress messages (default: TRUE)
#'
#' @return List containing:
#'   - tuned: List of tuned mlr3 learners (if model_tuning is "tuned" or "all")
#'   - untuned: List of untuned mlr3 learners (if model_tuning is "untuned" or "all")
#'
#' @importFrom sparklyr spark_connect spark_session_config spark_version sdf_copy_to spark_apply
#' @importFrom mlr3 TaskClassif lrn msr rsmp resample
#' @importFrom paradox ps p_int p_dbl generate_design_random
#' @importFrom jsonlite toJSON fromJSON
#' @importFrom DBI dbRemoveTable
#' @importFrom dplyr %>% collect
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Connect to Spark cluster
#' library(sparklyr)
#' config <- spark_config()
#' config$spark.task.cpus <- 8  # Each task claims 8 cores (1 full executor)
#' sc <- spark_connect(method = "databricks", config = config)
#'
#' # Prepare data
#' X_train <- iris[, 1:4]
#' Y_train <- ifelse(iris$Species == "setosa", 1, 0)
#'
#' # Define algorithms
#' algorithms <- list(
#'   ranger = list(
#'     learner = "classif.ranger",
#'     param_space = paradox::ps(
#'       num.trees = paradox::p_int(100, 500),
#'       mtry.ratio = paradox::p_dbl(0.1, 1),
#'       min.node.size = paradox::p_int(1, 10)
#'     ),
#'     measure = "classif.auc",
#'     predict_type = "prob"  # Required for classif.auc
#'   )
#' )
#'
#' # Train models
#' results <- auto_tune_classifier_spark(
#'   sc = sc,
#'   X_train = X_train,
#'   Y_train = Y_train,
#'   algorithms = algorithms,
#'   cv_folds = 5,
#'   n_evals = 50,
#'   model_tuning = "all",
#'   seed = 123,
#'   cores_to_use = 1
#' )
#'
#' # Access models
#' tuned_model <- results$tuned$ranger
#' untuned_model <- results$untuned$ranger
#'
#' # Make predictions
#' predictions <- tuned_model$predict_newdata(newdata = iris[1:10, 1:4])
#'
#' # Disconnect
#' spark_disconnect(sc)
#' }
#'
auto_tune_classifier_spark <- function(sc,
                                 X_train,
                                 Y_train,
                                 algorithms,
                                 cv_folds = 5,
                                 n_evals = 50,
                                 model_tuning = "all",
                                 seed = 123,
                                 cores_to_use = 1,
                                 verbose = TRUE) {

  # Set seed
  if (!is.null(seed)) set.seed(seed)

  # Basic validation
  X_train <- validate_feature_matrix(X_train, "X_train", allow_na = FALSE)
  Y_train <- validate_binary_outcome(Y_train, "Y_train")
  validate_matching_dimensions(X_train, Y_train, "X_train", "Y_train")

  if (!model_tuning %in% c("untuned", "tuned", "all")) {
    stop("model_tuning must be: untuned, tuned, or all")
  }

  # Set default predict_type
  for (algo_name in names(algorithms)) {
    if (!"predict_type" %in% names(algorithms[[algo_name]])) {
      algorithms[[algo_name]]$predict_type <- "prob"
    }
  }

  # Prepare training data
  train_data <- data.frame(X_train)
  train_data$target <- factor(Y_train, levels = c(0, 1), labels = .TARGET_LEVELS)
  
  # Create mlr3 task
  task <- TaskClassif$new(
    id = "training_task",
    backend = train_data,
    target = "target",
    positive = .TARGET_POSITIVE
  )

  # Train untuned models
  untuned_learners <- NULL

  if (model_tuning %in% c("untuned", "all")) {
    untuned_learners <- list()

    for (algo_name in names(algorithms)) {
      algo_spec <- algorithms[[algo_name]]
      learner <- create_learner(algo_spec$learner, algo_spec$predict_type, cores_to_use)
      learner$train(task)
      untuned_learners[[algo_name]] <- learner
    }
  }
  
  # Train tuned models
  tuned_learners <- NULL
  tuning_results <- NULL

  if (model_tuning %in% c("tuned", "all")) {
    tuned_learners <- list()
    tuning_results <- list()

    for (algo_name in names(algorithms)) {
      algo_spec <- algorithms[[algo_name]]

      # Generate random parameter combinations
      set.seed(seed)
      param_combos <- generate_param_combinations(algo_spec$param_space, n_evals)

      # Create Spark DataFrame of parameter combinations
      param_df <- data.frame(
        param_id = seq_len(n_evals),
        param_json = sapply(param_combos, toJSON, auto_unbox = TRUE),
        stringsAsFactors = FALSE
      )
      param_spark <- sdf_copy_to(sc, param_df, "params_temp",
                                  overwrite = TRUE,
                                  repartition = n_evals)
      
      # Distribute evaluation across executors using spark_apply
      eval_results <- param_spark %>%
        spark_apply(
          function(param_batch, context) {
            suppressPackageStartupMessages({
              library(mlr3)
              library(mlr3learners)
              library(jsonlite)
              library(lgr)
            })

            # Suppress mlr3 output
            lgr::get_logger("mlr3")$set_threshold("error")
            lgr::get_logger("bbotk")$set_threshold("error")

            # Process each parameter set
            results <- lapply(seq_len(nrow(param_batch)), function(i) {
              tryCatch({
                params <- fromJSON(param_batch$param_json[i], simplifyVector = TRUE)

                # Create task
                task <- TaskClassif$new(
                  id = "task",
                  backend = context$train_data,
                  target = "target",
                  positive = context$target_positive
                )

                # Create learner with parameters
                learner <- context$create_learner(context$learner_id, context$predict_type, context$cores_to_use)
                learner$param_set$values <- as.list(params)

                # Cross-validation
                resampling <- rsmp("cv", folds = context$cv_folds)
                rr <- resample(task, learner, resampling, store_models = FALSE)

                # Get score
                measure <- msr(context$measure_id)
                score <- rr$aggregate(measure)

                data.frame(
                  param_id = as.integer(param_batch$param_id[i]),
                  score = as.numeric(score),
                  param_json = as.character(param_batch$param_json[i]),
                  error = NA_character_,
                  stringsAsFactors = FALSE
                )
              }, error = function(e) {
                data.frame(
                  param_id = as.integer(param_batch$param_id[i]),
                  score = NA_real_,
                  param_json = as.character(param_batch$param_json[i]),
                  error = as.character(e$message),
                  stringsAsFactors = FALSE
                )
              })
            })

            do.call(rbind, results)
          },
          context = list(
            train_data = train_data,
            learner_id = algo_spec$learner,
            measure_id = algo_spec$measure,
            predict_type = algo_spec$predict_type,
            cv_folds = cv_folds,
            cores_to_use = cores_to_use,
            create_learner = create_learner,
            target_positive = .TARGET_POSITIVE
          )
        ) %>%
        collect()

      # Remove failed evaluations
      eval_results <- eval_results[!is.na(eval_results$score), ]

      if (nrow(eval_results) == 0) {
        stop(sprintf("All parameter evaluations failed for %s", algo_name))
      }

      # Find best parameters
      best_idx <- which.max(eval_results$score)
      best_params <- fromJSON(eval_results$param_json[best_idx])
      best_score <- eval_results$score[best_idx]

      # Train final model with best parameters
      final_learner <- create_learner(algo_spec$learner, algo_spec$predict_type, cores_to_use)
      final_learner$param_set$values <- as.list(best_params)
      final_learner$train(task)

      tuned_learners[[algo_name]] <- final_learner

      tuning_results[[algo_name]] <- list(
        best_params = best_params,
        best_score = best_score,
        measure = algo_spec$measure,
        archive = eval_results
      )

      # Clean up temporary table
      DBI::dbRemoveTable(sc, "params_temp")
    }
  }

  # Return results
  result <- list()

  if (model_tuning %in% c("tuned", "all")) {
    result$tuned <- tuned_learners
    result$tuning_results <- tuning_results
  }

  if (model_tuning %in% c("untuned", "all")) {
    result$untuned <- untuned_learners
  }

  return(result)
}


#' Generate Random Parameter Combinations from ParamSet
#'
#' @param param_space paradox ParamSet defining the search space
#' @param n Number of random combinations to generate
#' @return List of parameter lists
generate_param_combinations <- function(param_space, n) {
  design <- paradox::generate_design_random(param_space, n)
  param_list <- lapply(seq_len(nrow(design$data)), function(i) {
    as.list(design$data[i, ])
  })
  return(param_list)
}
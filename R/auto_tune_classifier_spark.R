#' Automatically Tune and Train Multiple Classification Models with Spark Distribution
#'
#' This function distributes hyperparameter search across Spark executors using spark_apply,
#' maximizing cluster utilization for large-scale tuning operations.
#'
#' @param sc Active Spark connection from spark_connect()
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
#' @param cores_to_use Number of cores to use per learner for threading (default: NULL). When NULL, uses cores_per_executor from cluster configuration.
#' @param verbose Logical indicating whether to print progress messages (default: TRUE)
#'
#' @return List containing:
#'   - tuned: List of tuned mlr3 learners (if model_tuning is "tuned" or "all")
#'   - untuned: List of untuned mlr3 learners (if model_tuning is "untuned" or "all")
#'   - tuning_results: List of tuning results for each algorithm (if model_tuning is "tuned" or "all")
#'   - cluster_info: Information about the cluster configuration
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
#' sc <- spark_connect(method = "databricks")
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
#'   cores_to_use = NULL  # Uses cores_per_executor by default
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
                                 cores_to_use = NULL,
                                 verbose = TRUE) {
  
  # Set seed for reproducibility if provided
  if (!is.null(seed)) {
    if (!is.numeric(seed) || length(seed) != 1) {
      stop("seed must be a single numeric value")
    }
    set.seed(seed)
    if (verbose) message(paste("Random seed set to:", seed))
  }

  # Input validation
  X_train <- validate_feature_matrix(X_train, "X_train", allow_na = FALSE)
  Y_train <- validate_binary_outcome(Y_train, "Y_train")
  validate_matching_dimensions(X_train, Y_train, "X_train", "Y_train")

  # Validate model_tuning parameter
  valid_tuning_options <- c("untuned", "tuned", "all")
  if (!model_tuning %in% valid_tuning_options) {
    stop(paste("model_tuning must be one of:", paste(valid_tuning_options, collapse = ", ")))
  }

  # Validate cv_folds
  if (!is.numeric(cv_folds) || length(cv_folds) != 1 || cv_folds < 2) {
    stop("cv_folds must be a single numeric value >= 2")
  }

  # Validate n_evals
  if (!is.numeric(n_evals) || length(n_evals) != 1 || n_evals < 1) {
    stop("n_evals must be a single positive numeric value")
  }

  # Validate algorithms
  if (!is.list(algorithms) || is.null(names(algorithms)) || length(algorithms) == 0) {
    stop("algorithms must be a named list with at least one algorithm specification")
  }

  required_fields <- c("learner", "measure")
  for (algo_name in names(algorithms)) {
    algo_spec <- algorithms[[algo_name]]
    if (!is.list(algo_spec) || !all(required_fields %in% names(algo_spec))) {
      stop(sprintf("Algorithm '%s' must contain 'learner' and 'measure' elements", algo_name))
    }

    # For tuned models, validate param_space is provided
    if (model_tuning %in% c("tuned", "all")) {
      if (!"param_space" %in% names(algo_spec)) {
        stop(sprintf("Algorithm '%s' missing 'param_space' field (required for tuned models)", algo_name))
      }
    }

    # Set default predict_type if not provided
    if (!"predict_type" %in% names(algo_spec)) {
      algorithms[[algo_name]]$predict_type <- "prob"
    }

    # Validate predict_type
    valid_predict_types <- c("prob", "response")
    if (!algo_spec$predict_type %in% valid_predict_types) {
      stop(sprintf("Algorithm '%s': predict_type must be one of: %s",
                  algo_name, paste(valid_predict_types, collapse = ", ")))
    }

    # Validate predict_type is compatible with measure
    prob_measures <- c("classif.auc", "classif.prauc", "classif.logloss", "classif.bbrier")
    measure_name <- algo_spec$measure

    if (measure_name %in% prob_measures && algo_spec$predict_type != "prob") {
      stop(sprintf("Algorithm '%s': measure '%s' requires predict_type = 'prob'",
                  algo_name, measure_name))
    }
  }
  
  # ========== CLUSTER DETECTION ==========
  cluster_info <- detect_cluster_config(sc, verbose)

  # Set cores_to_use if not specified
  if (is.null(cores_to_use)) {
    cores_to_use <- cluster_info$cores_per_executor
    if (verbose) message(sprintf("Setting cores_to_use to %d (cores per executor)", cores_to_use))
  } else {
    if (!is.numeric(cores_to_use) || length(cores_to_use) != 1 || cores_to_use < 1) {
      stop("cores_to_use must be a single positive numeric value")
    }
    if (verbose) message(sprintf("Using %d cores per learner", cores_to_use))
  }

  # ========== PREPARE DATA ==========
  # Prepare training data
  train_data <- data.frame(X_train)
  # Convert to factor with levels to match mlr3 expectations (consistent with non-Spark version)
  train_data$target <- factor(Y_train, levels = c(0, 1), labels = .TARGET_LEVELS)

  if (verbose) {
    message(sprintf("Training samples: %s, Features: %d",
                    format(nrow(train_data), big.mark = ","), ncol(X_train)))
  }
  
  # Create mlr3 task
  task <- TaskClassif$new(
    id = "training_task",
    backend = train_data,
    target = "target",
    positive = .TARGET_POSITIVE
  )
  
  # ========== TRAIN UNTUNED MODELS ==========
  untuned_learners <- NULL

  if (model_tuning %in% c("untuned", "all")) {
    if (verbose) message("Training untuned models...")

    untuned_learners <- list()

    for (algo_name in names(algorithms)) {
      if (verbose) message(paste("  Training", algo_name, "..."))

      algo_spec <- algorithms[[algo_name]]
      start_time <- Sys.time()

      # Create learner with default parameters and threading support
      learner <- create_learner(algo_spec$learner, algo_spec$predict_type, cores_to_use)

      # Train model
      learner$train(task)

      training_duration <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

      if (verbose) {
        message(sprintf("  Completed in %.2f seconds", training_duration))
      }

      untuned_learners[[algo_name]] <- learner
    }
  }
  
  # ========== TRAIN TUNED MODELS ==========
  tuned_learners <- NULL
  tuning_results <- NULL

  if (model_tuning %in% c("tuned", "all")) {
    # Broadcast training data to all executors
    train_broadcast <- sdf_copy_to(sc, train_data, "train_broadcast",
                                    overwrite = TRUE,
                                    repartition = cluster_info$total_cores)

    if (verbose) message("Training tuned models...")
    
    tuned_learners <- list()
    tuning_results <- list()
    
    for (algo_name in names(algorithms)) {
      if (verbose) message(paste("  Tuning", algo_name, "..."))

      algo_spec <- algorithms[[algo_name]]
      start_time <- Sys.time()

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
                                  repartition = cluster_info$total_cores)

      if (verbose) {
        message(sprintf("    Evaluating %d parameter sets (%d-fold CV) across %d cores",
                        n_evals, cv_folds, cluster_info$total_cores))
      }
      
      # Distribute evaluation across executors using spark_apply
      eval_results <- param_spark %>%
        spark_apply(
          function(param_batch, context) {
            # This code runs on Spark executors
            suppressPackageStartupMessages({
              library(mlr3)
              library(mlr3learners)
              library(jsonlite)
            })

            # Suppress mlr3 output
            lgr::get_logger("mlr3")$set_threshold("error")
            lgr::get_logger("bbotk")$set_threshold("error")

            # Get training data and constants from context
            train_data <- context$train_data
            TARGET_POSITIVE <- context$target_positive
            create_learner_func <- context$create_learner

            # Process each parameter set in this batch
            results <- lapply(seq_len(nrow(param_batch)), function(i) {
              tryCatch({
                param_json <- param_batch$param_json[i]
                param_id <- param_batch$param_id[i]

                # Parse parameters
                params <- fromJSON(param_json, simplifyVector = TRUE)

                # Create task
                task <- TaskClassif$new(
                  id = "task",
                  backend = train_data,
                  target = "target",
                  positive = TARGET_POSITIVE
                )

                # Create learner with threading support and parameters
                learner <- create_learner_func(context$learner_id, context$predict_type, context$cores_to_use)
                learner$param_set$values <- as.list(params)
                
                # Perform cross-validation
                resampling <- rsmp("cv", folds = context$cv_folds)
                rr <- resample(task, learner, resampling, store_models = FALSE)
                
                # Get score
                measure <- msr(context$measure_id)
                score <- rr$aggregate(measure)
                
                data.frame(
                  param_id = as.integer(param_id),
                  score = as.numeric(score),
                  param_json = as.character(param_json),
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
          ),
          columns = list(
            param_id = "integer",
            score = "double",
            param_json = "character",
            error = "character"
          ),
          packages = c("mlr3", "mlr3learners", "jsonlite", "lgr")
        ) %>%
        collect()
      
      end_time <- Sys.time()
      
      # Check for errors
      if (any(!is.na(eval_results$error))) {
        n_errors <- sum(!is.na(eval_results$error))
        if (verbose) {
          message(sprintf("    Warning: %d/%d parameter sets failed", n_errors, n_evals))
        }
      }

      # Remove failed evaluations
      eval_results <- eval_results[!is.na(eval_results$score), ]

      if (nrow(eval_results) == 0) {
        stop(sprintf("All parameter evaluations failed for %s", algo_name))
      }

      # Find best parameters
      best_idx <- which.max(eval_results$score)
      best_params <- fromJSON(eval_results$param_json[best_idx])
      best_score <- eval_results$score[best_idx]

      tuning_duration <- as.numeric(difftime(end_time, start_time, units = "mins"))

      if (verbose) {
        message(sprintf("    Best %s: %.4f (completed in %.2f minutes)",
                        algo_spec$measure, best_score, tuning_duration))
      }

      # Train final model with best parameters and threading support
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
    
    # Cleanup broadcast table
    DBI::dbRemoveTable(sc, "train_broadcast")
  }
  
  # ========== FORMAT RESULTS ==========
  result <- list()

  if (model_tuning %in% c("tuned", "all")) {
    result$tuned <- tuned_learners
    result$tuning_results <- tuning_results
  }

  if (model_tuning %in% c("untuned", "all")) {
    result$untuned <- untuned_learners
  }

  # Always include cluster_info for Spark version
  result$cluster_info <- cluster_info

  if (verbose) message("Model training completed successfully!")

  structure(result, class = "mlr3_ensemble")
}


#' Detect Spark Cluster Configuration
#'
#' @param sc Spark connection
#' @param verbose Print information
#' @return List with cluster configuration details
detect_cluster_config <- function(sc, verbose = TRUE) {
  
  tryCatch({
    # Get configuration
    conf <- spark_session_config(sc)
    
    # Try to get executor info
    n_executors <- NULL
    cores_per_executor <- NULL
    
    # Method 1: Configuration settings
    if ("spark.dynamicAllocation.maxExecutors" %in% names(conf)) {
      n_executors <- as.numeric(conf[["spark.dynamicAllocation.maxExecutors"]])
    } else if ("spark.executor.instances" %in% names(conf)) {
      n_executors <- as.numeric(conf[["spark.executor.instances"]])
    }
    
    if ("spark.executor.cores" %in% names(conf)) {
      cores_per_executor <- as.numeric(conf[["spark.executor.cores"]])
    }
    
    # Method 2: Query Spark directly (Databricks-specific)
    if (is.null(n_executors) || is.null(cores_per_executor)) {
      if (requireNamespace("SparkR", quietly = TRUE)) {
        tryCatch({
          spark_r_session <- SparkR::sparkR.session()
          executor_info <- SparkR::sql("SELECT * FROM spark_catalog.system.executors")
          executor_df <- SparkR::collect(executor_info)
          
          if (nrow(executor_df) > 0) {
            n_executors <- nrow(executor_df) - 1  # Exclude driver
            if (is.null(cores_per_executor) && "cores" %in% colnames(executor_df)) {
              cores_per_executor <- max(executor_df$cores[executor_df$id != "driver"])
            }
          }
        }, error = function(e) {
          # Silent fail, will use defaults
        })
      }
    }
    
    # Fallback defaults for typical Databricks clusters
    if (is.null(n_executors)) n_executors <- 8
    if (is.null(cores_per_executor)) cores_per_executor <- 8
    
    total_cores <- n_executors * cores_per_executor
    
    # Get memory info
    executor_memory <- conf[["spark.executor.memory"]]
    if (is.null(executor_memory)) executor_memory <- "Unknown"
    
    cluster_info <- list(
      n_executors = n_executors,
      cores_per_executor = cores_per_executor,
      total_cores = total_cores,
      executor_memory = executor_memory,
      spark_version = spark_version(sc)
    )

    if (verbose) {
      message(sprintf("Cluster: Spark %s, %d executors, %d cores/executor (%d total cores)",
                      cluster_info$spark_version, cluster_info$n_executors,
                      cluster_info$cores_per_executor, cluster_info$total_cores))
    }
    
    return(cluster_info)
    
  }, error = function(e) {
    warning(paste("Could not fully detect cluster config:", e$message))
    return(list(
      n_executors = 4,
      cores_per_executor = 4,
      total_cores = 16,
      executor_memory = "Unknown",
      spark_version = spark_version(sc)
    ))
  })
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


#' Print Method for mlr3_ensemble
#'
#' @param x mlr3_ensemble object
#' @param ... Additional arguments
#' @export
print.mlr3_ensemble <- function(x, ...) {
  cat("Distributed mlr3 Ensemble Classifier\n")
  cat("====================================\n\n")
  
  if (!is.null(x$tuned)) {
    cat("Tuned Models:", length(x$tuned), "\n")
    cat("Algorithms:", paste(names(x$tuned), collapse = ", "), "\n\n")
    
    if (!is.null(x$tuning_results)) {
      cat("Best CV Performance (Tuned):\n")
      for (algo_name in names(x$tuning_results)) {
        result <- x$tuning_results[[algo_name]]
        cat(sprintf("  %s (%s): %.4f\n", 
                    algo_name, result$measure, result$best_score))
      }
      cat("\n")
    }
  }
  
  if (!is.null(x$untuned)) {
    cat("Untuned Models:", length(x$untuned), "\n")
    cat("Algorithms:", paste(names(x$untuned), collapse = ", "), "\n\n")
  }
  
  cat("Cluster Configuration:\n")
  cat(sprintf("  Executors: %d\n", x$cluster_info$n_executors))
  cat(sprintf("  Cores per Executor: %d\n", x$cluster_info$cores_per_executor))
  cat(sprintf("  Total Cores: %d\n", x$cluster_info$total_cores))
  cat(sprintf("  Executor Memory: %s\n", x$cluster_info$executor_memory))
  
  invisible(x)
}

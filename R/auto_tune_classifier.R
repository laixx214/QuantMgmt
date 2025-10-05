#' Create Learner with Threading Support
#'
#' Helper function to create an mlr3 learner with appropriate threading
#' configuration based on the learner type.
#'
#' @param learner_id mlr3 learner ID (e.g., "classif.ranger", "classif.xgboost")
#' @param predict_type Prediction type ("prob" or "response")
#' @param cores Number of cores to use for threading
#' @return mlr3 learner with threading configured
#' @keywords internal
#' @noRd
create_learner <- function(learner_id, predict_type, cores) {
  learner <- lrn(learner_id, predict_type = predict_type)

  # Set threading parameters based on learner type
  if (grepl("ranger", learner_id, fixed = TRUE)) {
    learner$param_set$values$num.threads <- cores
  } else if (grepl("xgboost", learner_id, fixed = TRUE)) {
    learner$param_set$values$nthread <- cores
  }

  return(learner)
}

#' Automatically Tune Binary Classification Algorithms
#'
#' This function automatically tunes ranger (Random Forest) and/or xgboost algorithms
#' using specified search spaces and optimization criteria for binary classification.
#'
#' @param X_train Matrix or data frame of training features
#' @param Y_train Vector of training outcomes (binary classification: 0/1 or logical)
#' @param algorithms Named list of algorithms to tune. Each element should contain:
#'                  - learner: mlr3 learner ID (e.g., "classif.ranger", "classif.xgboost")
#'                  - param_space: paradox::ParamSet defining search space (optional if model_tuning = "untuned")
#'                  - measure: tuning criterion (e.g., "classif.prauc", "classif.auc")
#'                  - predict_type: prediction type - "prob" for probabilities or "response" for class labels (default: "prob")
#' @param cv_folds Number of cross-validation folds for tuning (default: 2)
#' @param n_evals Number of parameter evaluations for tuning (default: 15)
#' @param cores_to_use Number of cores to use for parallel processing (default: detectCores() - 1)
#' @param model_tuning Character indicating which models to return: "untuned", "tuned", or "all" (default: "all")
#' @param verbose Logical indicating whether to enable verbose output for parallel processing
#'                         diagnostics (default: TRUE). When TRUE, enables future.debug output.
#' @param seed Integer seed for reproducibility (default: 123). When set, ensures reproducible results.
#'
#' @return List containing (structure depends on model_tuning parameter):
#'   - tuned: List of tuned mlr3 learners (if model_tuning is "tuned" or "all")
#'   - untuned: List of untuned mlr3 learners (if model_tuning is "untuned" or "all")
#'
#' @importFrom future plan multisession sequential
#' @importFrom future.apply future_lapply
#' @importFrom mlr3 TaskClassif lrn msr rsmp
#' @importFrom mlr3tuning ti tnr
#' @importFrom bbotk trm
#' @importFrom paradox ps p_int p_dbl
#' @importFrom parallel detectCores
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Define search spaces with explicit learner specification
#' algorithms <- list(
#'   ranger = list(
#'     learner = "classif.ranger",
#'     param_space = paradox::ps(
#'       num.trees = paradox::p_int(100, 500),
#'       mtry.ratio = paradox::p_dbl(0.1, 1),
#'       min.node.size = paradox::p_int(1, 10)
#'     ),
#'     measure = "classif.prauc",
#'     predict_type = "prob"  # Required for classif.prauc
#'   ),
#'   xgboost = list(
#'     learner = "classif.xgboost",
#'     param_space = paradox::ps(
#'       nrounds = paradox::p_int(50, 200),
#'       eta = paradox::p_dbl(0.01, 0.3, logscale = TRUE),
#'       max_depth = paradox::p_int(3, 8)
#'     ),
#'     measure = "classif.auc",
#'     predict_type = "prob"  # Required for classif.auc
#'   )
#' )
#'
#' # Tune algorithms
#' results <- auto_tune_classifier(X_train, Y_train, algorithms, seed = 123)
#' }
#'
auto_tune_classifier <- function(X_train, Y_train, algorithms,
                              cv_folds = 2, n_evals = 15,
                              cores_to_use = max(1, detectCores() - 1),
                              model_tuning = "all",
                              verbose = TRUE,
                              seed = 123) {

    # Set seed for reproducibility if provided
    if (!is.null(seed)) {
        if (!is.numeric(seed) || length(seed) != 1) {
            stop("seed must be a single numeric value")
        }
        set.seed(seed)
        if (verbose) message(paste("Random seed set to:", seed))
    }

    # Enable verbose parallel diagnostics if requested
    if (verbose) {
        options(future.debug = TRUE)
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

    # Setup parallel processing for local execution
    if (verbose) message(paste("Using", cores_to_use, "cores for parallel processing"))

    # Create task
    task_data <- data.frame(X_train)
    # Convert to factor with levels to match mlr3 expectations
    task_data$target <- factor(Y_train, levels = c(0, 1), labels = .TARGET_LEVELS)

    task <- TaskClassif$new(
        id = "auto_tune_task",
        backend = task_data,
        target = "target",
        positive = .TARGET_POSITIVE
    )

    # Internal tuning function with local parallel processing
    tune_learner <- function(learner, param_space, task, measure_name) {
        measure <- msr(measure_name)

        # Setup future plan for local parallel processing
        # Limit workers to avoid oversubscription
        n_workers <- min(cores_to_use, n_evals)
        plan(multisession, workers = n_workers)

        # Create tuning instance
        instance <- ti(
            task = task,
            learner = learner,
            resampling = rsmp("cv", folds = cv_folds),
            measures = measure,
            search_space = param_space,
            terminator = trm("evals", n_evals = n_evals)
        )

        # Use random search (mlr3 automatically uses future backend if configured)
        tuner <- tnr("random_search")

        # Time the tuning process
        tuning_start <- Sys.time()
        tuner$optimize(instance)
        tuning_duration <- as.numeric(difftime(Sys.time(), tuning_start, units = "secs"))
        if (verbose) message(paste("  Tuning completed in", round(tuning_duration, 2), "seconds"))

        # Reset to sequential processing
        plan(sequential)

        learner$param_set$values <- instance$result_learner_param_vals
        return(learner)
    }

    # Initialize result lists
    tuned_learners <- list()
    untuned_learners <- list()

    # Process untuned learners if needed
    if (model_tuning %in% c("untuned", "all")) {
        if (verbose) message("Training untuned learners...")
        for (algo_name in names(algorithms)) {
            algo_spec <- algorithms[[algo_name]]
            if (verbose) message(paste("  Training", algo_name, "..."))

            # Create learner with default parameters
            learner_untuned <- create_learner(algo_spec$learner, algo_spec$predict_type, cores_to_use)

            # Train untuned learner with timing
            training_start <- Sys.time()
            learner_untuned$train(task)
            training_duration <- as.numeric(difftime(Sys.time(), training_start, units = "secs"))

            untuned_learners[[algo_name]] <- learner_untuned

            if (verbose) message(paste("  Completed in", round(training_duration, 2), "seconds"))
        }
    }

    # Process tuned learners if needed
    if (model_tuning %in% c("tuned", "all")) {
        if (verbose) message("Training tuned learners...")
        for (algo_name in names(algorithms)) {
            algo_spec <- algorithms[[algo_name]]
            if (verbose) message(paste("  Tuning", algo_name, "..."))

            # Create learner
            learner_tuned <- create_learner(algo_spec$learner, algo_spec$predict_type, cores_to_use)

            # Tune the learner (training happens within tuning process)
            learner_tuned <- tune_learner(learner_tuned, algo_spec$param_space, task, algo_spec$measure)

            # Train the tuned learner on the full task
            training_start <- Sys.time()
            learner_tuned$train(task)
            training_duration <- as.numeric(difftime(Sys.time(), training_start, units = "secs"))
            if (verbose) message(paste("  Final training completed in", round(training_duration, 2), "seconds"))

            tuned_learners[[algo_name]] <- learner_tuned
        }
    }

    # Return results based on model_tuning parameter
    result <- list()
    if (model_tuning %in% c("tuned", "all")) result$tuned <- tuned_learners
    if (model_tuning %in% c("untuned", "all")) result$untuned <- untuned_learners

    if (verbose) message("Model training completed successfully!")

    structure(result, class = "mlr3_ensemble")
}

#' Get Default Search Spaces for Common Algorithms
#'
#' Helper function to get default parameter search spaces for ranger and xgboost.
#' These can be used with any mlr3 learner that supports these parameters.
#'
#' @param algorithm Algorithm name ("ranger" or "xgboost")
#' @return ps() object with default search space
#'
#' @importFrom paradox ps p_int p_dbl
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Get default search spaces
#' ranger_space <- get_default_search_space("ranger")
#' xgboost_space <- get_default_search_space("xgboost")
#'
#' # Use in algorithms list
#' algorithms <- list(
#'   my_ranger = list(
#'     learner = "classif.ranger",
#'     param_space = ranger_space,
#'     measure = "classif.auc"
#'   )
#' )
#' }
#'
get_default_search_space <- function(algorithm) {
    if (algorithm == "ranger") {
        return(ps(
            num.trees = p_int(100, 500),
            mtry.ratio = p_dbl(0.1, 1),
            min.node.size = p_int(1, 10)
        ))
    } else if (algorithm == "xgboost") {
        return(ps(
            nrounds = p_int(50, 200),
            eta = p_dbl(0.01, 0.3, logscale = TRUE),
            max_depth = p_int(3, 8),
            subsample = p_dbl(0.5, 1),
            colsample_bytree = p_dbl(0.5, 1)
        ))
    } else {
        stop("Algorithm must be 'ranger' or 'xgboost'")
    }
}

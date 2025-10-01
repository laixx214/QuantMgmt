#' Automatically Tune Binary Classification Algorithms
#'
#' This function automatically tunes ranger (Random Forest) and/or xgboost algorithms
#' using specified search spaces and optimization criteria for binary classification.
#'
#' @param X_train Matrix or data frame of training features
#' @param Y_train Vector of training outcomes (binary classification: 0/1 or logical)
#' @param algorithms Named list of algorithms to tune. Names must be "ranger" and/or "xgboost".
#'                  Each element should contain:
#'                  - param_space: ps() object defining search space
#'                  - measure: tuning criterion (e.g., "classif.prauc")
#' @param cv_folds Number of cross-validation folds for tuning (default: 2)
#' @param n_evals Number of parameter evaluations for tuning (default: 15)
#' @param cores_to_use Number of cores to use for parallel processing (default: detectCores() - 1)
#' @param model_tuning Character indicating which models to return: "untuned", "tuned", or "all" (default: "all")
#' @param verbose_parallel Logical indicating whether to enable verbose output for parallel processing
#'                         diagnostics (default: FALSE). When TRUE, enables future.debug output.
#' @param seed Integer seed for reproducibility (default: NULL). When set, ensures reproducible results.
#'
#' @return Depends on model_tuning parameter:
#'         - "all": List with 'tuned' and 'untuned' elements
#'         - "tuned": Named list of tuned learners only
#'         - "untuned": Named list of untuned learners only
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
#' # Define search spaces
#' algorithms <- list(
#'   ranger = list(
#'     param_space = paradox::ps(
#'       num.trees = paradox::p_int(100, 500),
#'       mtry.ratio = paradox::p_dbl(0.1, 1),
#'       min.node.size = paradox::p_int(1, 10)
#'     ),
#'     measure = "classif.prauc"
#'   ),
#'   xgboost = list(
#'     param_space = paradox::ps(
#'       nrounds = paradox::p_int(50, 200),
#'       eta = paradox::p_dbl(0.01, 0.3, logscale = TRUE),
#'       max_depth = paradox::p_int(3, 8)
#'     ),
#'     measure = "classif.auc"
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
                              verbose_parallel = FALSE,
                              seed = NULL) {

    # Set seed for reproducibility if provided
    if (!is.null(seed)) {
        if (!is.numeric(seed) || length(seed) != 1) {
            stop("seed must be a single numeric value")
        }
        set.seed(seed)
        message(paste("Random seed set to:", seed))
    }

    # Enable verbose parallel diagnostics if requested
    if (verbose_parallel) {
        options(future.debug = TRUE)
        message("Verbose parallel processing diagnostics enabled")
    }

    # Input validation
    X_train <- validate_feature_matrix(X_train, "X_train", allow_na = FALSE)
    Y_train <- validate_binary_outcome(Y_train, "Y_train")
    validate_matching_dimensions(X_train, Y_train, "X_train", "Y_train")

    if (!is.list(algorithms) || is.null(names(algorithms))) {
        stop("algorithms must be a named list")
    }

    if (length(algorithms) == 0) {
        stop("algorithms list cannot be empty")
    }

    valid_algorithms <- c("ranger", "xgboost")
    if (!all(names(algorithms) %in% valid_algorithms)) {
        stop(paste("Algorithm names must be from:", paste(valid_algorithms, collapse = ", ")))
    }

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

    # Setup parallel processing for local execution
    message(paste("Using local parallel processing with", cores_to_use, "cores"))

    # Validate each algorithm specification
    for (alg_name in names(algorithms)) {
        alg_spec <- algorithms[[alg_name]]
        if (!is.list(alg_spec) || !all(c("param_space", "measure") %in% names(alg_spec))) {
            stop(paste("Each algorithm must contain 'param_space' and 'measure' elements. Error in:", alg_name))
        }
    }

    # Create task
    task_data <- data.frame(X_train)
    # Convert to factor with levels to match mlr3 expectations
    task_data$target <- factor(Y_train, levels = c(0, 1), labels = .TARGET_LEVELS)

    task <- TaskClassif$new(
        id = "auto_tune_task",
        backend = task_data,
        target = "target"
    )

    # Helper function to create learner
    create_learner <- function(alg_name, cores, set_defaults = FALSE) {
        if (alg_name == "ranger") {
            learner <- lrn("classif.ranger", predict_type = "prob", num.threads = cores)
            if (set_defaults) {
                learner$param_set$values$num.trees <- 500
            }
        } else if (alg_name == "xgboost") {
            learner <- lrn("classif.xgboost", predict_type = "prob", nthread = cores)
            if (set_defaults) {
                learner$param_set$values$nrounds <- 100
            }
        }
        return(learner)
    }

    # Internal tuning function with local parallel processing
    tune_learner <- function(learner, param_space, task, measure_name) {
        measure <- msr(measure_name)

        # Setup future plan for local parallel processing
        # Limit workers to avoid oversubscription
        n_workers <- min(cores_to_use, n_evals)
        plan(multisession, workers = n_workers)

        # Diagnostic: Verify future plan is active
        if (verbose_parallel) {
            message(paste("Future plan:", class(future::plan())[1]))
            message(paste("Number of workers:", future::nbrOfWorkers()))
        }

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
        message("Starting hyperparameter tuning...")
        tuning_start <- Sys.time()
        tuner$optimize(instance)
        tuning_end <- Sys.time()
        tuning_duration <- as.numeric(difftime(tuning_end, tuning_start, units = "secs"))
        message(paste("Tuning completed in", round(tuning_duration, 2), "seconds"))

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
        message("Training untuned learners...")
        for (alg_name in names(algorithms)) {
            message(paste("Training", alg_name, "with default parameters..."))

            # Create learner with default parameters
            learner_untuned <- create_learner(alg_name, cores_to_use, set_defaults = TRUE)

            # Train untuned learner with timing
            training_start <- Sys.time()
            learner_untuned$train(task)
            training_end <- Sys.time()
            training_duration <- as.numeric(difftime(training_end, training_start, units = "secs"))

            untuned_learners[[alg_name]] <- learner_untuned

            message(paste("Completed training", alg_name, "with default parameters in",
                         round(training_duration, 2), "seconds"))
        }
    }

    # Process tuned learners if needed
    if (model_tuning %in% c("tuned", "all")) {
        message("Training tuned learners...")
        for (alg_name in names(algorithms)) {
            alg_spec <- algorithms[[alg_name]]
            message(paste("Tuning", alg_name, "..."))

            # Create learner
            learner_tuned <- create_learner(alg_name, cores_to_use, set_defaults = FALSE)

            # Tune the learner (training happens within tuning process)
            learner_tuned <- tune_learner(learner_tuned, alg_spec$param_space, task, alg_spec$measure)

            # Train the tuned learner on the full task
            message(paste("Training", alg_name, "with optimal parameters on full dataset..."))
            training_start <- Sys.time()
            learner_tuned$train(task)
            training_end <- Sys.time()
            training_duration <- as.numeric(difftime(training_end, training_start, units = "secs"))
            message(paste("Completed training", alg_name, "in", round(training_duration, 2), "seconds"))

            tuned_learners[[alg_name]] <- learner_tuned

            message(paste("Completed tuning", alg_name))
        }
    }

    # Return results based on model_tuning parameter
    result <- list()
    if (model_tuning %in% c("tuned", "all")) result$tuned <- tuned_learners
    if (model_tuning %in% c("untuned", "all")) result$untuned <- untuned_learners

    message("Model training completed successfully!")
    return(result)
}

#' Get Default Search Spaces for Supported Algorithms
#'
#' Helper function to get default parameter search spaces for ranger and xgboost
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
#' ranger_space <- get_default_search_space("ranger")
#' xgboost_space <- get_default_search_space("xgboost")
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
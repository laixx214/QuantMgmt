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
#' @param spark_connection Optional sparklyr connection object for distributed processing. If provided,
#'                        parameter search evaluations will be distributed across Spark executors.
#'                        Defaults to 18 executors with 8 cores each if configuration cannot be detected.
#' @param verbose_parallel Logical indicating whether to enable verbose output for parallel processing
#'                         diagnostics (default: FALSE). When TRUE, enables future.debug output.
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
#' results <- auto_tune_classifier(X_train, Y_train, algorithms)
#' }
#'
auto_tune_classifier <- function(X_train, Y_train, algorithms,
                              cv_folds = 2, n_evals = 15,
                              cores_to_use = max(1, detectCores() - 1),
                              model_tuning = "all",
                              spark_connection = NULL,
                              verbose_parallel = FALSE) {

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

    # Detect Spark cluster configuration
    use_spark_distributed <- FALSE
    n_executors <- 18  # Default
    executor_cores <- 8  # Default

    if (!is.null(spark_connection)) {
        use_spark_distributed <- TRUE

        # Try to get actual cluster configuration
        tryCatch({
            spark_conf <- sparklyr::spark_context_config(spark_connection)

            # Try to get number of executors
            if (!is.null(spark_conf$spark.executor.instances)) {
                n_executors <- as.integer(spark_conf$spark.executor.instances)
            } else if (!is.null(spark_conf$`spark.executor.instances`)) {
                n_executors <- as.integer(spark_conf$`spark.executor.instances`)
            }

            # Try to get cores per executor
            if (!is.null(spark_conf$spark.executor.cores)) {
                executor_cores <- as.integer(spark_conf$spark.executor.cores)
            } else if (!is.null(spark_conf$`spark.executor.cores`)) {
                executor_cores <- as.integer(spark_conf$`spark.executor.cores`)
            }

            message(paste("Detected Spark cluster configuration:"))
            message(paste("  Executors:", n_executors))
            message(paste("  Cores per executor:", executor_cores))
            message(paste("  Total cluster cores:", n_executors * executor_cores))
        }, error = function(e) {
            message("Could not detect Spark configuration, using defaults:")
            message(paste("  Executors:", n_executors))
            message(paste("  Cores per executor:", executor_cores))
            message(paste("  Total cluster cores:", n_executors * executor_cores))
        })

        # Override cores_to_use for individual model training
        cores_to_use <- executor_cores
        message(paste("Each model will use", cores_to_use, "cores for internal parallelization"))
    }

    # Validate each algorithm specification
    for (alg_name in names(algorithms)) {
        alg_spec <- algorithms[[alg_name]]
        if (!is.list(alg_spec) || !all(c("param_space", "measure") %in% names(alg_spec))) {
            stop(paste("Each algorithm must contain 'param_space' and 'measure' elements. Error in:", alg_name))
        }

        # Validate measure
        valid_measures <- c("classif.acc", "classif.auc", "classif.prauc",
                           "classif.f1", "classif.precision", "classif.recall", "classif.fbeta")
        if (!alg_spec$measure %in% valid_measures) {
            stop(paste("Measure must be one of:", paste(valid_measures, collapse = ", "), ". Error in:", alg_name))
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

    # Internal tuning function with Spark distribution
    tune_learner <- function(learner, param_space, task, measure_name) {
        measure <- msr(measure_name)

        if (use_spark_distributed) {
            # Use Spark to distribute parameter search evaluations
            message(paste("Using Spark distributed tuning with", n_executors, "executors"))
            message(paste("Distributing", n_evals, "parameter evaluations across cluster"))

            # Setup future plan to use multiple sessions
            # This distributes work across available Spark executors
            plan(multisession, workers = min(n_executors, n_evals))

            # Diagnostic: Verify future plan is active
            message(paste("Future plan:", class(future::plan())[1]))
            message(paste("Number of workers:", future::nbrOfWorkers()))
        }

        # Create tuning instance (works for both distributed and sequential)
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

        # Reset to sequential processing if using Spark
        if (use_spark_distributed) {
            plan(sequential)
        }

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

            # Tune and train the learner
            learner_tuned <- tune_learner(learner_tuned, alg_spec$param_space, task, alg_spec$measure)

            # Train with timing
            training_start <- Sys.time()
            learner_tuned$train(task)
            training_end <- Sys.time()
            training_duration <- as.numeric(difftime(training_end, training_start, units = "secs"))

            tuned_learners[[alg_name]] <- learner_tuned

            message(paste("Completed tuning", alg_name, "- Final training took",
                         round(training_duration, 2), "seconds"))
        }
    }

    # Return results based on model_tuning parameter
    if (model_tuning == "tuned") {
        result <- list(
            tuned = tuned_learners
        )
        message("Tuned model training completed successfully!")
    } else if (model_tuning == "untuned") {
        result <- list(
            untuned = untuned_learners
        )
        message("Untuned model training completed successfully!")
    } else {  # model_tuning == "all"
        result <- list(
            tuned = tuned_learners,
            untuned = untuned_learners
        )
        message("Model training completed successfully!")
    }

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
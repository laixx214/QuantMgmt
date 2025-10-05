#' Evaluate Auto-Tuned Classifier Performance
#'
#' This function evaluates the performance of models returned by auto_tune_classifier
#' or auto_tune_classifier_spark using comprehensive binary classification metrics.
#'
#' @param model_results Output from auto_tune_classifier or auto_tune_classifier_spark
#'                      function (list or named list of learners/models)
#' @param data List with two elements: X_validate (features) and Y_validate (outcomes).
#'             If NULL (default), uses training data from the learners.
#'             Note: For Spark models, data cannot be NULL - validation data must be provided.
#' @param decision_threshold Numeric threshold for converting probabilities to class predictions (default: 0.5)
#'
#' @return Data frame with performance metrics for each model, including improvement calculations
#'         when both tuned and untuned models are present
#'
#' @importFrom dplyr %>% collect pull
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Evaluate with validation data
#' validation_data <- list(X_validate = X_val, Y_validate = Y_val)
#' results <- evaluate_classifier_performance(model_results, validation_data)
#'
#' # Evaluate with training data
#' results <- evaluate_classifier_performance(model_results)
#' }
#'
evaluate_classifier_performance <- function(model_results, data = NULL, decision_threshold = 0.5) {
    # Input validation
    if (!is.list(model_results)) {
        stop("model_results must be a list")
    }

    if (!is.null(data)) {
        if (!is.list(data) || length(data) != 2 ||
            !all(c("X_validate", "Y_validate") %in% names(data))) {
            stop("data must be a list with elements 'X_validate' and 'Y_validate'")
        }
    }

    # Check if Spark models require validation data
    is_spark <- inherits(model_results, "spark_ml_ensemble")
    if (is_spark && is.null(data)) {
        stop("For Spark models, data cannot be NULL. Please provide validation data with X_validate and Y_validate.")
    }

    # Get predictions using predict_classifier
    X_prediction <- if (!is.null(data)) data$X_validate else NULL
    predictions <- predict_classifier(model_results, X_prediction)

    # Get true labels
    if (!is.null(data)) {
        Y_eval <- data$Y_validate

        # If Y_validate is a Spark DataFrame, collect to local
        if (inherits(Y_eval, "tbl_spark")) {
            # Assume Y_validate is a single column Spark DataFrame
            Y_eval <- Y_eval %>% dplyr::collect() %>% dplyr::pull(1)
        }

        # Convert to binary numeric if needed
        if (!is.numeric(Y_eval)) {
            Y_eval <- as.numeric(as.character(Y_eval) == .TARGET_POSITIVE)
        }

        data_source <- "validation"
    } else {
        # Extract true labels from training data (use any available learner's task)
        if (!is.null(predictions$tuned_prediction)) {
            # Get from tuned models
            first_learner <- NULL
            if ("tuned" %in% names(model_results) && length(model_results$tuned) > 0) {
                first_learner <- model_results$tuned[[1]]
            } else if (inherits(model_results[[1]], "Learner")) {
                first_learner <- model_results[[1]]
            }
        } else if (!is.null(predictions$untuned_prediction)) {
            # Get from untuned models
            if ("untuned" %in% names(model_results) && length(model_results$untuned) > 0) {
                first_learner <- model_results$untuned[[1]]
            }
        }

        if (is.null(first_learner)) {
            stop("Could not extract training labels from model_results")
        }

        Y_eval <- as.numeric(as.character(first_learner$task$truth()) == .TARGET_POSITIVE)
        data_source <- "training"
    }

    # Helper function to process predictions for a given model type
    process_predictions <- function(prediction_list, model_type, Y_eval, decision_threshold, data_source) {
        results <- list()

        for (alg_name in names(prediction_list)) {
            if (alg_name != "ensemble_avg") {
                predicted_prob <- prediction_list[[alg_name]]
                metrics <- binary_classification_metrics(predicted_prob, Y_eval, decision_threshold)

                # Add model information
                metrics$algorithm <- alg_name
                metrics$model_type <- model_type
                metrics$data_source <- data_source

                # Store results
                result_name <- paste(alg_name, model_type, sep = "_")
                results[[result_name]] <- metrics
            }
        }

        # Add ensemble results if available
        if ("ensemble_avg" %in% names(prediction_list)) {
            predicted_prob <- prediction_list$ensemble_avg
            metrics <- binary_classification_metrics(predicted_prob, Y_eval, decision_threshold)

            metrics$algorithm <- "ensemble"
            metrics$model_type <- model_type
            metrics$data_source <- data_source

            results[[paste("ensemble", model_type, sep = "_")]] <- metrics
        }

        return(results)
    }

    # Initialize results list
    all_results <- list()

    # Process untuned predictions if available
    if (!is.null(predictions$untuned_prediction)) {
        all_results <- c(all_results, process_predictions(
            predictions$untuned_prediction, "untuned", Y_eval, decision_threshold, data_source
        ))
    }

    # Process tuned predictions if available
    if (!is.null(predictions$tuned_prediction)) {
        all_results <- c(all_results, process_predictions(
            predictions$tuned_prediction, "tuned", Y_eval, decision_threshold, data_source
        ))
    }

    # Combine all results
    result_df <- do.call(rbind, all_results)
    rownames(result_df) <- names(all_results)

    # Calculate improvements if both tuned and untuned exist
    has_tuned <- !is.null(predictions$tuned_prediction)
    has_untuned <- !is.null(predictions$untuned_prediction)

    if (has_tuned && has_untuned) {
        improvement_results <- list()

        # Get algorithm names from both prediction types
        tuned_algs <- names(predictions$tuned_prediction)[names(predictions$tuned_prediction) != "ensemble_avg"]
        untuned_algs <- names(predictions$untuned_prediction)[names(predictions$untuned_prediction) != "ensemble_avg"]
        common_algs <- intersect(tuned_algs, untuned_algs)

        for (alg_name in common_algs) {
            tuned_row <- paste(alg_name, "tuned", sep = "_")
            untuned_row <- paste(alg_name, "untuned", sep = "_")

            if (tuned_row %in% rownames(result_df) && untuned_row %in% rownames(result_df)) {
                tuned_metrics <- result_df[tuned_row, ]
                untuned_metrics <- result_df[untuned_row, ]

                # Calculate percentage improvements using vectorized approach
                metric_names <- c("classif.acc", "classif.auc", "classif.prauc",
                                 "classif.f1", "classif.precision", "classif.recall")

                improvement_values <- sapply(metric_names, function(metric) {
                    untuned_val <- untuned_metrics[[metric]]
                    tuned_val <- tuned_metrics[[metric]]
                    if (is.na(untuned_val) || untuned_val == 0) {
                        return(NA)
                    }
                    ((tuned_val - untuned_val) / untuned_val) * 100
                })

                improvement <- data.frame(
                    as.list(improvement_values),
                    algorithm = alg_name,
                    model_type = "improvement_pct",
                    data_source = tuned_metrics$data_source,
                    stringsAsFactors = FALSE
                )

                improvement_name <- paste(alg_name, "improvement_pct", sep = "_")
                improvement_results[[improvement_name]] <- improvement
            }
        }

        if (length(improvement_results) > 0) {
            improvement_df <- do.call(rbind, improvement_results)
            rownames(improvement_df) <- names(improvement_results)
            result_df <- rbind(result_df, improvement_df)
        }
    }

    # Reorder columns for better readability
    col_order <- c("algorithm", "model_type", "data_source",
                   "classif.acc", "classif.auc", "classif.prauc",
                   "classif.f1", "classif.precision", "classif.recall")
    result_df <- result_df[, col_order]

    return(result_df)
}
#' Evaluate Auto-Tuned Classifier Performance
#'
#' This function evaluates the performance of models returned by auto_tune_classifier
#' using comprehensive binary classification metrics.
#'
#' @param model_results Output from auto_tune_classifier function (list or named list of learners)
#' @param data List with two elements: X_validate (features) and Y_validate (outcomes).
#'             If NULL (default), uses training data from the learners.
#' @param decision_threshold Numeric threshold for converting probabilities to class predictions (default: 0.5)
#'
#' @return Data frame with performance metrics for each model, including improvement calculations
#'         when both tuned and untuned models are present
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

    # Get predictions using predict_classifier
    X_prediction <- if (!is.null(data)) data$X_validate else NULL
    predictions <- predict_classifier(model_results, X_prediction)

    # Get true labels
    if (!is.null(data)) {
        Y_eval <- data$Y_validate
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

        Y_eval <- as.numeric(as.character(first_learner$task$truth()) == "TRUE")
        data_source <- "training"
    }

    # Initialize results list
    all_results <- list()

    # Process untuned predictions if available
    if (!is.null(predictions$untuned_prediction)) {
        for (alg_name in names(predictions$untuned_prediction)) {
            if (alg_name != "ensemble_avg") {
                predicted_prob <- predictions$untuned_prediction[[alg_name]]
                metrics <- binary_classification_metrics(predicted_prob, Y_eval, decision_threshold)

                # Add model information
                metrics$algorithm <- alg_name
                metrics$model_type <- "untuned"
                metrics$data_source <- data_source

                # Store results
                result_name <- paste(alg_name, "untuned", sep = "_")
                all_results[[result_name]] <- metrics
            }
        }

        # Add ensemble results for untuned
        if ("ensemble_avg" %in% names(predictions$untuned_prediction)) {
            predicted_prob <- predictions$untuned_prediction$ensemble_avg
            metrics <- binary_classification_metrics(predicted_prob, Y_eval, decision_threshold)

            metrics$algorithm <- "ensemble"
            metrics$model_type <- "untuned"
            metrics$data_source <- data_source

            all_results[["ensemble_untuned"]] <- metrics
        }
    }

    # Process tuned predictions if available
    if (!is.null(predictions$tuned_prediction)) {
        for (alg_name in names(predictions$tuned_prediction)) {
            if (alg_name != "ensemble_avg") {
                predicted_prob <- predictions$tuned_prediction[[alg_name]]
                metrics <- binary_classification_metrics(predicted_prob, Y_eval, decision_threshold)

                # Add model information
                metrics$algorithm <- alg_name
                metrics$model_type <- "tuned"
                metrics$data_source <- data_source

                # Store results
                result_name <- paste(alg_name, "tuned", sep = "_")
                all_results[[result_name]] <- metrics
            }
        }

        # Add ensemble results for tuned
        if ("ensemble_avg" %in% names(predictions$tuned_prediction)) {
            predicted_prob <- predictions$tuned_prediction$ensemble_avg
            metrics <- binary_classification_metrics(predicted_prob, Y_eval, decision_threshold)

            metrics$algorithm <- "ensemble"
            metrics$model_type <- "tuned"
            metrics$data_source <- data_source

            all_results[["ensemble_tuned"]] <- metrics
        }
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

                # Calculate percentage improvements
                improvement <- data.frame(
                    classif.acc = ifelse(is.na(untuned_metrics$classif.acc) || untuned_metrics$classif.acc == 0, NA,
                                       ((tuned_metrics$classif.acc - untuned_metrics$classif.acc) / untuned_metrics$classif.acc) * 100),
                    classif.auc = ifelse(is.na(untuned_metrics$classif.auc) || untuned_metrics$classif.auc == 0, NA,
                                       ((tuned_metrics$classif.auc - untuned_metrics$classif.auc) / untuned_metrics$classif.auc) * 100),
                    classif.prauc = ifelse(is.na(untuned_metrics$classif.prauc) || untuned_metrics$classif.prauc == 0, NA,
                                         ((tuned_metrics$classif.prauc - untuned_metrics$classif.prauc) / untuned_metrics$classif.prauc) * 100),
                    classif.f1 = ifelse(is.na(untuned_metrics$classif.f1) || untuned_metrics$classif.f1 == 0, NA,
                                      ((tuned_metrics$classif.f1 - untuned_metrics$classif.f1) / untuned_metrics$classif.f1) * 100),
                    classif.precision = ifelse(is.na(untuned_metrics$classif.precision) || untuned_metrics$classif.precision == 0, NA,
                                             ((tuned_metrics$classif.precision - untuned_metrics$classif.precision) / untuned_metrics$classif.precision) * 100),
                    classif.recall = ifelse(is.na(untuned_metrics$classif.recall) || untuned_metrics$classif.recall == 0, NA,
                                          ((tuned_metrics$classif.recall - untuned_metrics$classif.recall) / untuned_metrics$classif.recall) * 100),
                    algorithm = alg_name,
                    model_type = "improvement_pct",
                    data_source = tuned_metrics$data_source
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
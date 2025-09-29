#' Predict Using Auto-Tuned Classifiers
#'
#' This function generates predictions using models returned by auto_tune_classifier.
#' Returns probability predictions for individual models and ensemble predictions.
#'
#' @param model_results Output from auto_tune_classifier function (list or named list of learners)
#' @param X_prediction Matrix or data frame of features for prediction.
#'                     If NULL (default), uses training data from the learners.
#'
#' @return List with two elements:
#'         - 'untuned_prediction': Individual algorithm predictions + ensemble from untuned models (NULL if no untuned models)
#'         - 'tuned_prediction': Individual algorithm predictions + ensemble from tuned models (NULL if no tuned models)
#'         Each element contains probability predictions for each algorithm and ensemble averages
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Predict on new data
#' predictions <- predict_classifier(model_results, X_new)
#'
#' # Predict on training data
#' predictions <- predict_classifier(model_results)
#' }
#'
predict_classifier <- function(model_results, X_prediction = NULL) {
    # Input validation
    if (!is.list(model_results)) {
        stop("model_results must be a list")
    }

    # Detect structure of model_results
    has_tuned <- FALSE
    has_untuned <- FALSE
    model_list <- list()

    if ("tuned" %in% names(model_results) && "untuned" %in% names(model_results)) {
        # Structure from model_tuning = "all"
        has_tuned <- length(model_results$tuned) > 0
        has_untuned <- length(model_results$untuned) > 0
        if (has_tuned) model_list$tuned <- model_results$tuned
        if (has_untuned) model_list$untuned <- model_results$untuned
    } else if ("tuned" %in% names(model_results)) {
        # Structure from model_tuning = "tuned" (wrapped in list)
        has_tuned <- TRUE
        model_list$tuned <- model_results$tuned
    } else if ("untuned" %in% names(model_results)) {
        # Structure from model_tuning = "untuned" (wrapped in list)
        has_untuned <- TRUE
        model_list$untuned <- model_results$untuned
    } else {
        # Direct list of learners (from model_tuning = "tuned" or "untuned" without wrapper)
        # Check if these are mlr3 learners
        first_element <- model_results[[1]]
        if (inherits(first_element, "Learner")) {
            # Assume these are tuned models if they have been trained
            has_tuned <- TRUE
            model_list$tuned <- model_results
        } else {
            stop("Unrecognized structure in model_results")
        }
    }

    # Helper function to make predictions for a set of learners
    predict_learners <- function(learners, X_pred = NULL) {
        predictions <- list()

        for (alg_name in names(learners)) {
            learner <- learners[[alg_name]]

            if (is.null(X_pred)) {
                # Predict on training data
                pred_result <- learner$predict(learner$task)
                predictions[[alg_name]] <- pred_result$prob[, "TRUE"]
            } else {
                # Create prediction task for new data
                pred_data <- data.frame(X_pred)
                # Create dummy target (will be ignored for prediction)
                pred_data$target <- factor(rep("FALSE", nrow(pred_data)),
                                         levels = c("FALSE", "TRUE"))

                pred_task <- mlr3::TaskClassif$new(
                    id = "pred_task",
                    backend = pred_data,
                    target = "target"
                )

                pred_result <- learner$predict(pred_task)
                predictions[[alg_name]] <- pred_result$prob[, "TRUE"]
            }
        }

        return(predictions)
    }

    # Helper function to create ensemble predictions
    create_ensemble <- function(predictions_list) {
        if (length(predictions_list) == 0) return(NULL)

        # Convert to matrix for easier calculation
        pred_matrix <- do.call(cbind, predictions_list)

        # Calculate ensemble as simple average
        ensemble_avg <- rowMeans(pred_matrix, na.rm = TRUE)

        # Add ensemble to the list
        predictions_list$ensemble_avg <- ensemble_avg

        return(predictions_list)
    }

    # Initialize results with fixed structure
    result <- list(
        untuned_prediction = NULL,
        tuned_prediction = NULL
    )

    # Process untuned models if they exist
    if (has_untuned) {
        message("Generating predictions from untuned models...")
        untuned_predictions <- predict_learners(model_list$untuned, X_prediction)
        result$untuned_prediction <- create_ensemble(untuned_predictions)
    }

    # Process tuned models if they exist
    if (has_tuned) {
        message("Generating predictions from tuned models...")
        tuned_predictions <- predict_learners(model_list$tuned, X_prediction)
        result$tuned_prediction <- create_ensemble(tuned_predictions)
    }

    message("Prediction completed successfully!")
    return(result)
}
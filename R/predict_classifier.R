#' Predict Using Auto-Tuned Classifiers
#'
#' This function generates predictions using models returned by auto_tune_classifier
#' or auto_tune_classifier_spark. Returns probability predictions for individual
#' models and ensemble predictions.
#'
#' @param model_results Output from auto_tune_classifier or auto_tune_classifier_spark
#'                      function (list or named list of learners/models)
#' @param X_prediction Matrix or data frame of features for prediction.
#'                     If NULL (default), uses training data from the learners.
#'                     For Spark models, can also be a Spark DataFrame.
#'
#' @return List with two elements:
#'         - 'untuned_prediction': Individual algorithm predictions + ensemble from untuned models (NULL if no untuned models)
#'         - 'tuned_prediction': Individual algorithm predictions + ensemble from tuned models (NULL if no tuned models)
#'         Each element contains probability predictions for each algorithm and ensemble averages
#'
#' @importFrom mlr3 TaskClassif
#' @importFrom sparklyr ml_predict sdf_copy_to spark_connection
#' @importFrom dplyr %>% select mutate pull
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

    # Helper function to detect and extract model structure
    detect_model_structure <- function(model_results) {
        structure <- list(has_tuned = FALSE, has_untuned = FALSE, model_list = list())

        # Check for explicit tuned/untuned structure
        if ("tuned" %in% names(model_results) && "untuned" %in% names(model_results)) {
            # Structure from model_tuning = "all"
            structure$has_tuned <- length(model_results$tuned) > 0
            structure$has_untuned <- length(model_results$untuned) > 0
            if (structure$has_tuned) structure$model_list$tuned <- model_results$tuned
            if (structure$has_untuned) structure$model_list$untuned <- model_results$untuned
            return(structure)
        }

        # Check for single type with explicit structure
        if ("tuned" %in% names(model_results)) {
            structure$has_tuned <- TRUE
            structure$model_list$tuned <- model_results$tuned
            return(structure)
        }

        if ("untuned" %in% names(model_results)) {
            structure$has_untuned <- TRUE
            structure$model_list$untuned <- model_results$untuned
            return(structure)
        }

        # Assume direct list of learners
        if (length(model_results) > 0 && inherits(model_results[[1]], "Learner")) {
            structure$has_tuned <- TRUE
            structure$model_list$tuned <- model_results
            return(structure)
        }

        stop("Unrecognized structure in model_results")
    }

    # Detect structure of model_results
    structure <- detect_model_structure(model_results)
    has_tuned <- structure$has_tuned
    has_untuned <- structure$has_untuned
    model_list <- structure$model_list

    # Detect if models are Spark ML models
    is_spark <- inherits(model_results, "spark_ml_ensemble")

    # Helper function to make predictions for a set of learners (mlr3 or Spark)
    predict_learners <- function(learners, X_pred = NULL, is_spark = FALSE) {
        predictions <- list()

        for (alg_name in names(learners)) {
            learner <- learners[[alg_name]]

            if (is_spark) {
                # Spark ML model prediction
                if (is.null(X_pred)) {
                    stop("X_prediction cannot be NULL for Spark models. Training data predictions not supported.")
                }

                # Check if X_pred is already a Spark DataFrame
                if (inherits(X_pred, "tbl_spark")) {
                    pred_data <- X_pred
                } else {
                    # Convert to Spark DataFrame
                    sc <- spark_connection(learner)
                    pred_data <- sdf_copy_to(sc, as.data.frame(X_pred), "pred_temp", overwrite = TRUE)
                }

                # Get predictions from Spark model
                pred_result <- ml_predict(learner, pred_data)

                # Extract probability for positive class
                # Spark ML stores probabilities in a vector column, extract second element (positive class)
                predictions[[alg_name]] <- pred_result %>%
                    dplyr::select(probability) %>%
                    dplyr::mutate(prob_positive = probability[2]) %>%
                    dplyr::pull(prob_positive)

            } else {
                # mlr3 Learner prediction
                if (is.null(X_pred)) {
                    # Predict on training data
                    pred_result <- learner$predict(learner$task)
                    predictions[[alg_name]] <- pred_result$prob[, .TARGET_POSITIVE]
                } else {
                    # Create prediction task for new data
                    pred_data <- data.frame(X_pred)
                    # Create dummy target (will be ignored for prediction)
                    pred_data$target <- factor(rep(.TARGET_NEGATIVE, nrow(pred_data)),
                                             levels = .TARGET_LEVELS)

                    pred_task <- TaskClassif$new(
                        id = "pred_task",
                        backend = pred_data,
                        target = "target"
                    )

                    pred_result <- learner$predict(pred_task)
                    predictions[[alg_name]] <- pred_result$prob[, .TARGET_POSITIVE]
                }
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
        untuned_predictions <- predict_learners(model_list$untuned, X_prediction, is_spark)
        result$untuned_prediction <- create_ensemble(untuned_predictions)
    }

    # Process tuned models if they exist
    if (has_tuned) {
        message("Generating predictions from tuned models...")
        tuned_predictions <- predict_learners(model_list$tuned, X_prediction, is_spark)
        result$tuned_prediction <- create_ensemble(tuned_predictions)
    }

    message("Prediction completed successfully!")
    return(result)
}
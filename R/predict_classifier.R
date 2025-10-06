#' Predict Using Auto-Tuned Classifiers
#'
#' This function generates predictions using models returned by auto_tune_classifier
#' or auto_tune_classifier_spark. Returns probability predictions for individual
#' models and ensemble predictions.
#'
#' @param model_results Output from auto_tune_classifier or auto_tune_classifier_spark.
#'                      Can be a list of mlr3 Learner objects, or an S3 object of class
#'                      "spark_ml_ensemble" containing Spark ML models.
#' @param X_prediction Matrix, data frame, or Spark DataFrame of features for prediction.
#'                     - For mlr3 models: data.frame or matrix. If NULL, uses training data.
#'                     - For Spark models: data.frame or Spark DataFrame (tbl_spark). Cannot be NULL.
#'
#' @return List with two elements (structure depends on model_tuning used during training):
#'         - $untuned_prediction: Named list of probability vectors for each algorithm plus $ensemble_avg (NULL if no untuned models)
#'         - $tuned_prediction: Named list of probability vectors for each algorithm plus $ensemble_avg (NULL if no tuned models)
#'         Probability vectors represent the predicted probability of the positive class (1) for each observation.
#'
#' @importFrom mlr3 TaskClassif
#' @importFrom sparklyr ml_predict sdf_copy_to spark_connection spark_connection_find ft_vector_assembler
#' @importFrom dplyr %>% select mutate pull collect
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Example 1: mlr3 models - predict on new data
#' predictions <- predict_classifier(model_results, X_new)
#'
#' # Example 2: mlr3 models - predict on training data
#' predictions <- predict_classifier(model_results)
#'
#' # Example 3: Spark models - predict on new data (data.frame)
#' predictions <- predict_classifier(spark_results, X_new)
#'
#' # Example 4: Spark models - predict on Spark DataFrame
#' new_data_tbl <- copy_to(sc, X_new, "new_data")
#' predictions <- predict_classifier(spark_results, new_data_tbl)
#'
#' # Access individual model predictions
#' rf_probs <- predictions$tuned_prediction$rf
#' ensemble_probs <- predictions$tuned_prediction$ensemble_avg
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

                # Apply feature assembler if pred_data doesn't have 'features' column
                if (!"features" %in% colnames(pred_data)) {
                    feature_cols <- setdiff(colnames(pred_data), c("target", "label", "features", "prediction", "probability"))
                    pred_data <- pred_data %>% ft_vector_assembler(input_cols = feature_cols, output_col = "features")
                }

                # Get predictions from Spark model
                pred_result <- ml_predict(learner, pred_data)

                # Extract probability for positive class
                # Spark ML stores probabilities in a vector column
                # We need to use spark_apply or extract via UDF
                # For now, collect the data and extract locally
                pred_local <- pred_result %>% dplyr::collect()

                # Extract probability of positive class (second element of probability vector)
                # The probability column is a list-column where each element is a vector
                predictions[[alg_name]] <- sapply(pred_local$probability, function(x) x[2])

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
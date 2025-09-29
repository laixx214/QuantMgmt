#' Calculate Binary Classification Metrics
#'
#' This function calculates comprehensive performance metrics for binary classification
#' based on predicted probabilities and observed outcomes.
#'
#' @param predicted_prob Numeric vector of predicted probabilities (0 to 1) for the positive class
#' @param observed_y Logical or numeric vector of observed outcomes. Must be logical or contain only 0 and 1 values
#' @param decision_threshold Numeric threshold for converting probabilities to class predictions (default: 0.5)
#'
#' @return Data frame with classification metrics: classif.acc, classif.auc, classif.prauc,
#'         classif.f1, classif.precision, classif.recall
#'
#' @export
#'
#' @examples
#' # Example usage
#' predicted_prob <- c(0.1, 0.4, 0.35, 0.8, 0.65, 0.9, 0.2, 0.7, 0.85, 0.3)
#' observed_y <- c(0, 0, 0, 1, 1, 1, 0, 1, 1, 0)
#' binary_classification_metrics(predicted_prob, observed_y, decision_threshold = 0.5)
#'
binary_classification_metrics <- function(predicted_prob, observed_y, decision_threshold = 0.5) {
    # Input validation for observed_y
    if (is.logical(observed_y)) {
        actual <- as.numeric(observed_y)
    } else if (is.numeric(observed_y)) {
        if (!all(observed_y %in% c(0, 1))) {
            stop("observed_y must be logical or contain only values 0 and 1")
        }
        actual <- observed_y
    } else {
        stop("observed_y must be logical or numeric with values 0 and 1 only. Factors are not allowed.")
    }

    # Input validation for predicted_prob
    if (!is.numeric(predicted_prob)) {
        stop("predicted_prob must be numeric")
    }
    if (any(predicted_prob < 0) || any(predicted_prob > 1)) {
        stop("predicted_prob must be between 0 and 1")
    }
    if (length(predicted_prob) != length(actual)) {
        stop("predicted_prob and observed_y must have the same length")
    }

    # Input validation for decision_threshold
    if (!is.numeric(decision_threshold) || length(decision_threshold) != 1) {
        stop("decision_threshold must be a single numeric value")
    }
    if (decision_threshold < 0 || decision_threshold > 1) {
        stop("decision_threshold must be between 0 and 1")
    }

    # Use the provided decision threshold
    predicted_class <- as.numeric(predicted_prob > decision_threshold)

    # Confusion matrix components
    tp <- sum(actual == 1 & predicted_class == 1)
    tn <- sum(actual == 0 & predicted_class == 0)
    fp <- sum(actual == 0 & predicted_class == 1)
    fn <- sum(actual == 1 & predicted_class == 0)

    # Calculate metrics manually - return NA when denominators are 0
    accuracy <- (tp + tn) / (tp + tn + fp + fn)
    precision <- ifelse(tp + fp == 0, NA, tp / (tp + fp))
    recall <- ifelse(tp + fn == 0, NA, tp / (tp + fn))
    f1 <- ifelse(is.na(precision) || is.na(recall) || precision + recall == 0, NA, 2 * precision * recall / (precision + recall))

    # ROC-AUC calculation
    if(requireNamespace("pROC", quietly = TRUE)) {
        auc_val <- pROC::auc(actual, predicted_prob, quiet = TRUE)
    } else {
        auc_val <- NA
        warning("pROC package not available. AUC will be NA.")
    }

    # PR-AUC calculation using precrec
    if(requireNamespace("precrec", quietly = TRUE)) {
        tryCatch({
            pr_curve <- precrec::evalmod(scores = predicted_prob, labels = actual)
            auc_results <- precrec::auc(pr_curve)
            pr_auc_val <- auc_results$aucs[auc_results$curvetypes == "PRC"]
        }, error = function(e) {
            pr_auc_val <- NA
            warning("Error calculating PR-AUC: ", e$message)
        })
    } else {
        pr_auc_val <- NA
        warning("precrec package not available. PR-AUC will be NA.")
    }

    # Return results as data frame
    result <- data.frame(
        classif.acc = accuracy,
        classif.auc = as.numeric(auc_val),
        classif.prauc = as.numeric(pr_auc_val),
        classif.f1 = f1,
        classif.precision = precision,
        classif.recall = recall
    )

    return(result)
}
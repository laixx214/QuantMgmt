#' Input Validation Utilities
#'
#' @description Internal validation functions used across the package
#' @keywords internal
#' @noRd

#' Validate binary outcome variable
#'
#' @param y Vector of outcomes
#' @param var_name Name of variable for error messages
#' @return Numeric vector (0/1)
#' @keywords internal
#' @noRd
validate_binary_outcome <- function(y, var_name = "outcome") {
    if (is.logical(y)) {
        return(as.numeric(y))
    }

    if (is.numeric(y)) {
        if (any(is.na(y))) {
            stop(paste(var_name, "contains NA values"))
        }
        if (!all(y %in% c(0, 1))) {
            stop(paste(var_name, "must be logical or contain only values 0 and 1"))
        }
        return(y)
    }

    stop(paste(var_name, "must be logical or numeric with values 0 and 1 only. Factors are not allowed."))
}

#' Validate feature matrix
#'
#' @param X Feature matrix or data frame
#' @param var_name Name of variable for error messages
#' @param allow_na Whether to allow NA values (default: FALSE)
#' @return Validated matrix/data frame
#' @keywords internal
#' @noRd
validate_feature_matrix <- function(X, var_name = "features", allow_na = FALSE) {
    if (!is.matrix(X) && !is.data.frame(X)) {
        stop(paste(var_name, "must be a matrix or data frame"))
    }

    if (!allow_na && any(is.na(X))) {
        stop(paste(var_name, "contains NA values"))
    }

    if (any(sapply(X, function(col) any(is.infinite(col))))) {
        stop(paste(var_name, "contains infinite values"))
    }

    return(X)
}

#' Validate probability vector
#'
#' @param prob Probability vector
#' @param var_name Name of variable for error messages
#' @return Validated probability vector
#' @keywords internal
#' @noRd
validate_probabilities <- function(prob, var_name = "predicted_prob") {
    if (!is.numeric(prob)) {
        stop(paste(var_name, "must be numeric"))
    }

    if (any(is.na(prob))) {
        stop(paste(var_name, "contains NA values"))
    }

    if (any(prob < 0) || any(prob > 1)) {
        stop(paste(var_name, "must be between 0 and 1"))
    }

    return(prob)
}

#' Validate matching dimensions
#'
#' @param X Feature matrix
#' @param y Outcome vector
#' @param X_name Name of X for error messages
#' @param y_name Name of y for error messages
#' @keywords internal
#' @noRd
validate_matching_dimensions <- function(X, y, X_name = "X", y_name = "y") {
    if (nrow(X) != length(y)) {
        stop(paste(X_name, "and", y_name, "must have the same number of observations.",
                   "Got", nrow(X), "and", length(y)))
    }
}

#' Validate feature consistency between train and prediction
#'
#' @param X_train Training feature matrix
#' @param X_pred Prediction feature matrix
#' @keywords internal
#' @noRd
validate_feature_consistency <- function(X_train, X_pred) {
    if (ncol(X_train) != ncol(X_pred)) {
        stop(paste("Feature dimension mismatch: training data has", ncol(X_train),
                   "features but prediction data has", ncol(X_pred), "features"))
    }

    if (is.data.frame(X_train) && is.data.frame(X_pred)) {
        train_names <- names(X_train)
        pred_names <- names(X_pred)

        if (!all(pred_names %in% train_names)) {
            missing <- setdiff(pred_names, train_names)
            stop(paste("Prediction data contains features not in training data:",
                       paste(missing, collapse = ", ")))
        }

        if (!all(train_names %in% pred_names)) {
            missing <- setdiff(train_names, pred_names)
            stop(paste("Prediction data is missing features from training data:",
                       paste(missing, collapse = ", ")))
        }

        # Reorder prediction features to match training order
        if (!identical(train_names, pred_names)) {
            warning("Reordering prediction features to match training data order")
            X_pred <- X_pred[, train_names, drop = FALSE]
        }
    }

    return(X_pred)
}
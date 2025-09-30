#' Package Constants
#'
#' @description Internal constants used across the package
#' @keywords internal
#' @noRd

# Factor levels for binary classification targets
# Used consistently across all functions to ensure compatibility
.TARGET_LEVELS <- c("FALSE", "TRUE")
.TARGET_NEGATIVE <- "FALSE"
.TARGET_POSITIVE <- "TRUE"
#' Learner Utility Functions
#'
#' @description Internal utility functions for creating and configuring mlr3 learners
#' @keywords internal
#' @noRd
NULL

#' Create Learner with Threading Support
#'
#' Helper function to create an mlr3 learner with appropriate threading
#' configuration based on the learner type.
#'
#' @param learner_id mlr3 learner ID (e.g., "classif.ranger", "classif.xgboost")
#' @param predict_type Prediction type ("prob" or "response")
#' @param cores Number of cores to use for threading
#' @return mlr3 learner with threading configured
#' @keywords internal
#' @noRd
create_learner <- function(learner_id, predict_type, cores) {
  learner <- lrn(learner_id, predict_type = predict_type)

  # Set threading parameters based on learner type
  if (grepl("ranger", learner_id, fixed = TRUE)) {
    learner$param_set$values$num.threads <- cores
  } else if (grepl("xgboost", learner_id, fixed = TRUE)) {
    learner$param_set$values$nthread <- cores
  }

  return(learner)
}

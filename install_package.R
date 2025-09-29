# Install QuantMgmt Package
#
# This script installs the QuantMgmt package and its dependencies

# Install dependencies if not already installed
required_packages <- c(
    "devtools", "mlr3", "mlr3learners", "mlr3tuning",
    "bbotk", "ranger", "xgboost", "paradox",
    "pROC", "precrec", "parallel"
)

# Check which packages are missing
missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

if (length(missing_packages)) {
    cat("Installing missing dependencies:", paste(missing_packages, collapse = ", "), "\n")
    install.packages(missing_packages)
}

# Install QuantMgmt package
if (requireNamespace("devtools", quietly = TRUE)) {
    cat("Installing QuantMgmt package...\n")
    devtools::install("/Users/yufeng.lai/Documents/remote/QuantMgmt",
                      dependencies = TRUE,
                      upgrade = "never")
    cat("QuantMgmt package installed successfully!\n")
} else {
    stop("devtools package is required to install QuantMgmt")
}

# Test installation
tryCatch({
    library(QuantMgmt)
    cat("QuantMgmt package loaded successfully!\n")
    cat("Available functions:\n")
    cat("- auto_tune_classifier()\n")
    cat("- predict_classifier()\n")
    cat("- evaluate_classifier_performance()\n")
    cat("- binary_classification_metrics()\n")
    cat("- get_default_search_space()\n")
}, error = function(e) {
    cat("Error loading QuantMgmt package:", e$message, "\n")
})
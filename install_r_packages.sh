#!/bin/bash
# Databricks cluster init script to install R packages
# Installs all required packages for churn prediction workflows

set -e

# Install R packages
Rscript -e "
# Set CRAN mirror
options(repos = c(CRAN = 'https://cran.rstudio.com/'))

# List of required packages from pull_churn_data.r and run_prediction.r
required_packages <- c(
  'sparklyr',
  'dplyr',
  'DBI',
  'devtools',
  # QuantMgmt dependencies (Imports)
  'mlr3',
  'mlr3tuning',
  'bbotk',
  'paradox',
  'precrec',
  'parallel',
  'future',
  'future.apply',
  # QuantMgmt dependencies (Suggests)
  'mlr3learners',
  'ranger',
  'xgboost'
)

# Get already installed packages
installed <- installed.packages()[,'Package']

# Filter packages that need to be installed
to_install <- required_packages[!(required_packages %in% installed)]

if(length(to_install) > 0) {
  cat('Installing', length(to_install), 'packages...\\n')
  cat('Packages to install:', paste(to_install, collapse = ', '), '\\n')
  install.packages(to_install, dependencies = TRUE, Ncpus = parallel::detectCores())
  cat('Successfully installed', length(to_install), 'packages\\n')
} else {
  cat('All required packages already installed\\n')
}
"

echo "R package installation completed successfully!"
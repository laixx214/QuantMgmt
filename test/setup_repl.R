# === SETUP FUNCTION (run locally before db_repl) ===
setup_repl <- function() {
  cat("\n═══════════════════════════════════════════════════════════\n")
  cat("  QuantMgmt Test Suite - Setup\n")
  cat("═══════════════════════════════════════════════════════════\n\n")

  # Install dependencies
  cat("Installing dependencies...\n")
  pkgs <- c("brickster", "huxtable", "sparklyr", "dplyr", "devtools")
  new_pkgs <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
  if (length(new_pkgs) > 0) {
    install.packages(new_pkgs, repos = "https://cloud.r-project.org")
  }

  library(brickster)

  # Load environment variables
  cat("Loading Databricks configuration...\n")
  config_dir <- "/Users/yufeng.lai/Documents/remote/consumption_progression"
  env_file <- file.path(config_dir, ".databricks", ".databricks.env")

  if (!file.exists(env_file)) {
    stop("Config file not found: ", env_file)
  }

  env_lines <- readLines(env_file, warn = FALSE)
  extract <- function(lines, pattern) {
    line <- lines[grepl(pattern, lines)]
    if (length(line) == 0) return(NA)
    sub(paste0("^", pattern, "="), "", line[1])
  }

  host <- extract(env_lines, "DATABRICKS_HOST")
  cluster_id <- extract(env_lines, "DATABRICKS_CLUSTER_ID")
  token <- sub(".*token=([^;]+).*", "\\1", env_lines[grepl("token=", env_lines)])

  Sys.setenv(
    DATABRICKS_HOST = host,
    DATABRICKS_TOKEN = token,
    DATABRICKS_CLUSTER_ID = cluster_id
  )

  cat("✅ Environment configured\n")
  cat("   Host:", host, "\n")
  cat("   Cluster:", cluster_id, "\n\n")

  cat("Connect now? (y/n): ")
  response <- readline()

  if (tolower(trimws(response)) == "y") {
    cat("\nConnecting to Databricks REPL...\n")
    cat("After connecting, run these commands to test:\n")
    cat("  source('https://raw.githubusercontent.com/laixx214/QuantMgmt/refs/heads/main/test/run.R')\n")
    cat("  run_tests()\n\n")

    db_repl(cluster_id = cluster_id)
  } else {
    cat("\nTo connect later, run:\n")
    cat("  db_repl(cluster_id = '", cluster_id, "')\n\n", sep = "")
    cat("Then source and run tests:\n")
    cat("  source('https://raw.githubusercontent.com/laixx214/QuantMgmt/refs/heads/main/test/run.R')\n")
    cat("  run_tests()\n\n")
  }
}
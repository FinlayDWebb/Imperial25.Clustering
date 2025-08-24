# Total.Pipe.r
# =============================================
# MASTER PIPELINE SCRIPT (REVISED)
# =============================================
# Coordinates the entire workflow for a single dataset:
# 1. Processes one specific dataset
# 2. Runs imputation pipeline (Main.Pipe.r)
# 3. Runs clustering evaluation (Cluster.Pipe.r)
# =============================================

# Load required libraries
library(readr)
library(dplyr)
library(tools)
library(arrow) # Add this for Feather support

# Source the component scripts
source("Main.Pipe.r")
source("Cluster.Pipe.r")

# ----------------------------
# 0. CREATE OUTPUT DIRECTORIES
# ----------------------------

if (!dir.exists("imputation_results")) {
  dir.create("imputation_results")
}
if (!dir.exists("clustering_results")) {
  dir.create("clustering_results")
}

if (!dir.exists("imputed_datasets")) {
  dir.create("imputed_datasets")
}

# ----------------------------
# 1. DATASET CONFIGURATION
# ----------------------------

# Specify the exact dataset to process
dataset <- "Processed.Data/ty_wine.white_data.feather"

# Verify the dataset exists
if (!file.exists(dataset)) {
  stop("Error: Dataset file not found: ", dataset)
}

dataset_name <- file_path_sans_ext(basename(dataset))

cat("Processing dataset:", dataset, "\n")

# Define parameters for the pipeline
missing_rates <- c(0.10) # , 0.10, 0.15)
methods <- c("MICE", "FAMD", "missForest")
n_clusters <- c(2, 3, 5)  # Number of clusters for evaluation

# ----------------------------
# 2. PIPELINE EXECUTION
# ----------------------------

cat("\n\n", rep("=", 60), "\n", sep="")
cat("STARTING PIPELINE FOR DATASET:", dataset, "\n")
cat(rep("=", 60), "\n\n", sep="")


# ----------------------------
# A. RUN IMPUTATION PIPELINE
# ----------------------------
cat(">>> RUNNING IMPUTATION PIPELINE\n")
imputation_results <- run_imputation_pipeline(
  data_path = dataset,
  missing_rates = missing_rates,
  methods = methods
)

# Save results in imputation_results folder
imputation_file <- file.path("imputation_results", 
                                paste0(dataset_name, "_*_", missing_rates[1], "_imputed.feather"))
arrow::write_feather(imputation_results, imputation_file)
cat("Saved imputation results to:", imputation_file, "\n")

# ----------------------------
# B. RUN CLUSTERING EVALUATION
# ----------------------------
cat("\n>>> RUNNING CLUSTERING EVALUATION\n")

# Generate file pattern for current dataset's imputed files
clustering_results_list <- list()
for (k in n_clusters) {
  cat("\n>>> RUNNING CLUSTERING EVALUATION FOR", k, "CLUSTERS\n")
  imputed_pattern <- file.path("imputed_datasets", 
                          paste0(dataset_name, "_*_", missing_rates[1], "_imputed.feather"))
  
  clustering_results <- evaluate_clustering_performance(
    original_data_path = dataset,
    imputed_files_pattern = imputed_pattern,
    n_clusters = k,
  )
  
  # Save clustering results to clustering_results folder
  clustering_file <- file.path("clustering_results", 
                               paste0(dataset_name, "_", missing_rates[1], "_clustering_", k, "_results.feather"))
  arrow::write_feather(clustering_results, clustering_file)
  cat("Saved clustering results to:", clustering_file, "\n")
  
  # Store with cluster count as key
  clustering_results_list[[as.character(k)]] <- clustering_results
}

# ----------------------------
# 3. FINAL REPORT
# ----------------------------
cat("\n\n", rep("=", 60), "\n", sep="")
cat("PIPELINE EXECUTION COMPLETE\n")
cat(rep("=", 60), "\n\n", sep="")

cat("Processed dataset:", dataset, "\n")

cat("\nSummary of results files:\n")
cat("- Imputation results: imputation_results/", dataset_name, "_imputation_results.feather\n", sep="")
for (k in n_clusters) {
  cat("- Clustering results (", k, " clusters): clustering_results/", dataset_name, "_clustering_", k, "_results.feather\n", sep="")
}

cat("\n=== TOTAL PIPELINE EXECUTION COMPLETE ===\n")
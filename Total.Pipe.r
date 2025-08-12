# Total.Pipe.r
# =============================================
# MASTER PIPELINE SCRIPT (REVISED)
# =============================================
# Coordinates the entire workflow:
# 1. Iterates through multiple datasets
# 2. Runs imputation pipeline (Main.Pipe.r)
# 3. Runs clustering evaluation (Cluster.Pipe.r)
# 4. Consolidates all results
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

# Create directories for organized output
if (!dir.exists("imputation_results")) {
  dir.create("imputation_results")
}
if (!dir.exists("clustering_results")) {
  dir.create("clustering_results")
}

# ----------------------------
# 1. DATASET CONFIGURATION
# ----------------------------

# Get all Feather files from the Processed.Data folder
data_folder <- "Processed.Data"

if (!dir.exists(data_folder)) {
  stop("Error: 'Processed.Data' folder not found in current directory!")
}

datasets <- list.files(
  path = data_folder,
  pattern = "\\.feather$",
  full.names = TRUE,
  ignore.case = TRUE
)

if (length(datasets) == 0) {
  stop("Error: No Feather files found in 'Processed.Data' folder!")
}

cat("Found", length(datasets), "Feather files in", data_folder, ":\n")
for (i in seq_along(datasets)) {
  cat(sprintf("  %d. %s\n", i, basename(datasets[i])))
}

# Define parameters for the pipeline (unchanged)
missing_rates <- c(0.05) #,0.10, 0.15)
methods <- c("MICE", "FAMD", "missForest")
n_clusters <- 2  # Number of clusters for evaluation, stick with 2 for simplicity.

# ----------------------------
# 2. PIPELINE EXECUTION
# ----------------------------

# Initialize results storage
all_imputation_results <- list()
all_clustering_results <- list()

for (dataset in datasets) {
  dataset_name <- file_path_sans_ext(basename(dataset))

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

  # Save and store results in imputation_results folder
  imputation_file <- file.path("imputation_results", paste0(dataset_name, "_imputation_results.feather"))
  arrow::write_feather(imputation_results, imputation_file)
  all_imputation_results[[dataset_name]] <- imputation_results
  cat("Saved imputation results to:", imputation_file, "\n")
  
  # ----------------------------
  # B. RUN CLUSTERING EVALUATION
  # ----------------------------
  cat("\n>>> RUNNING CLUSTERING EVALUATION\n")

  # Generate file pattern for current dataset's imputed files
  imputed_pattern <- paste0(dataset_name, "_*_imputed.feather")
  
  clustering_results <- evaluate_clustering_performance(
    original_data_path = dataset,
    imputed_files_pattern = imputed_pattern,
    n_clusters = n_clusters,
    output_file = file.path("clustering_results", paste0(dataset_name, "_clustering_results.feather"))
  )
  
  # Store results
  all_clustering_results[[dataset_name]] <- clustering_results
  cat("Clustering evaluation complete for", dataset, "\n")
}

# ----------------------------
# 3. RESULTS CONSOLIDATION
# ----------------------------
cat("\n\n", rep("=", 60), "\n", sep="")
cat("CONSOLIDATING FINAL RESULTS\n")
cat(rep("=", 60), "\n\n", sep="")

# Combine all imputation results
combined_imputation <- bind_rows(all_imputation_results, .id = "Dataset")
arrow::write_feather(combined_imputation, "combined_imputation_results.feather")
cat("Saved combined imputation results: combined_imputation_results.feather\n")

# Combine all clustering results
combined_clustering <- bind_rows(all_clustering_results, .id = "Dataset")
arrow::write_feather(combined_clustering, "combined_clustering_results.feather")
cat("Saved combined clustering results: combined_clustering_results.feather\n")

# ----------------------------
# 4. FINAL REPORT
# ----------------------------
cat("\n\n", rep("=", 60), "\n", sep="")
cat("PIPELINE EXECUTION COMPLETE\n")
cat(rep("=", 60), "\n\n", sep="")

cat("Processed", length(datasets), "datasets:\n")
cat(paste("-", datasets), sep = "\n")

cat("\nSummary of results files:\n")
cat("- Imputation results for each dataset: imputation_results/[dataset]_imputation_results.feather\n")
cat("- Clustering results for each dataset: clustering_results/[dataset]_clustering_results.feather\n")
cat("- Combined imputation results: combined_imputation_results.feather\n")
cat("- Combined clustering results: combined_clustering_results.feather\n")

cat("\n=== TOTAL PIPELINE EXECUTION COMPLETE ===\n")
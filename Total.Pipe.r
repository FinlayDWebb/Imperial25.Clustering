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

# Get all CSV files from the Processed.Data folder
data_folder <- "Processed.Data"

# Check if folder exists
if (!dir.exists(data_folder)) {
  stop("Error: 'Processed.Data' folder not found in current directory!")
}

# Get all CSV files from the folder
datasets <- list.files(
  path = data_folder,
  pattern = "\\.csv$",
  full.names = TRUE,  # This gives full path like "Processed.Data/file1.csv"
  ignore.case = TRUE
)

# Check if any CSV files were found
if (length(datasets) == 0) {
  stop("Error: No CSV files found in 'Processed.Data' folder!")
}

# Print what files were found
cat("Found", length(datasets), "CSV files in", data_folder, ":\n")
for (i in seq_along(datasets)) {
  cat(sprintf("  %d. %s\n", i, basename(datasets[i])))
}

# Define parameters for the pipeline (unchanged)
missing_rates <- c(0.05) # , 0.10, 0.15
methods <- c("MICE", "FAMD", "missForest", "MIDAS")
n_clusters <- 2  # Number of clusters for evaluation, stick with 2 for simplicity.

# ----------------------------
# 2. PIPELINE EXECUTION
# ----------------------------

# Initialize results storage
all_imputation_results <- list()
all_clustering_results <- list()

for (dataset in datasets) {
  dataset_name <- file_path_sans_ext(basename(dataset))

  # Construct metadata path (assumes metadata files are named dataset_name.meta.csv)
  metadata_path <- file.path(data_folder, paste0(dataset_name, ".meta.csv"))
  
  # Verify metadata exists
  if (!file.exists(metadata_path)) {
    stop("Metadata file not found: ", metadata_path)
  }

  cat("\n\n", rep("=", 60), "\n", sep="")
  cat("STARTING PIPELINE FOR DATASET:", dataset, "\n")
  cat(rep("=", 60), "\n\n", sep="")
  
  # ----------------------------
  # A. RUN IMPUTATION PIPELINE
  # ----------------------------
  cat(">>> RUNNING IMPUTATION PIPELINE\n")
  imputation_results <- run_imputation_pipeline(
    data_path = dataset,
    metadata_path = metadata_path,  # Pass metadata path
    missing_rates = missing_rates,
    methods = methods
  )
  
  # Save and store results in imputation_results folder
  imputation_file <- file.path("imputation_results", paste0(dataset_name, "_imputation_results.csv"))
  write_csv(imputation_results, imputation_file)
  all_imputation_results[[dataset_name]] <- imputation_results
  cat("Saved imputation results to:", imputation_file, "\n")
  
  # ----------------------------
  # B. RUN CLUSTERING EVALUATION
  # ----------------------------
  cat("\n>>> RUNNING CLUSTERING EVALUATION\n")
  
  # Generate file pattern for current dataset's imputed files
  imputed_pattern <- paste0(dataset_name, "_*.csv")
  
  clustering_results <- evaluate_clustering_performance(
    original_data_path = dataset,
    metadata_path = metadata_path,  # Pass metadata path  
    imputed_files_pattern = imputed_pattern,
    n_clusters = n_clusters,
    output_file = file.path("clustering_results", paste0(dataset_name, "_clustering_results.csv"))
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
write_csv(combined_imputation, "combined_imputation_results.csv")
cat("Saved combined imputation results: combined_imputation_results.csv\n")

# Combine all clustering results
combined_clustering <- bind_rows(all_clustering_results, .id = "Dataset")
write_csv(combined_clustering, "combined_clustering_results.csv")
cat("Saved combined clustering results: combined_clustering_results.csv\n")

# ----------------------------
# 4. FINAL REPORT
# ----------------------------
cat("\n\n", rep("=", 60), "\n", sep="")
cat("PIPELINE EXECUTION COMPLETE\n")
cat(rep("=", 60), "\n\n", sep="")

cat("Processed", length(datasets), "datasets:\n")
cat(paste("-", datasets), sep = "\n")

cat("\nSummary of results files:\n")
cat("- Imputation results for each dataset: imputation_results/[dataset]_imputation_results.csv\n")
cat("- Clustering results for each dataset: clustering_results/[dataset]_clustering_results.csv\n")
cat("- Combined imputation results: combined_imputation_results.csv\n")
cat("- Combined clustering results: combined_clustering_results.csv\n")

cat("\n=== TOTAL PIPELINE EXECUTION COMPLETE ===\n")
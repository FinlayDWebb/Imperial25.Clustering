# Total.Pipe.r
# =============================================
# MASTER PIPELINE SCRIPT
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
# 1. DATASET CONFIGURATION
# ----------------------------

# Define datasets to process (add/remove as needed)
datasets <- c(
  "diagnosis_data.csv"
)

# Define parameters for the pipeline
missing_rates <- c(0.05, 0.10, 0.15)
methods <- c("MICE", "FAMD", "missForest", "MIDAS")
n_clusters <- 3  # Number of clusters for evaluation

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
  
  # Save and store results
  imputation_file <- paste0(dataset_name, "_imputation_results.csv")
  write_csv(imputation_results, imputation_file)
  all_imputation_results[[dataset_name]] <- imputation_results
  cat("Saved imputation results to:", imputation_file, "\n")
  
  # ----------------------------
  # B. RUN CLUSTERING EVALUATION
  # ----------------------------
  cat("\n>>> RUNNING CLUSTERING EVALUATION\n")
  
  # Generate file pattern for imputed datasets
  imputed_pattern <- paste0(dataset_name, "_*.csv")
  
  clustering_results <- evaluate_clustering_performance(
    original_data_path = dataset,
    imputed_files_pattern = imputed_pattern,
    n_clusters = n_clusters,
    output_file = paste0(dataset_name, "_clustering_results.csv")
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
cat("- Imputation results for each dataset: [dataset]_imputation_results.csv\n")
cat("- Clustering results for each dataset: [dataset]_clustering_results.csv\n")
cat("- Combined imputation results: combined_imputation_results.csv\n")
cat("- Combined clustering results: combined_clustering_results.csv\n")

cat("\n=== TOTAL PIPELINE EXECUTION COMPLETE ===\n")
# =============================================
# CLUSTERING-ONLY EVALUATION SCRIPT
# =============================================
# Steps 5-7: Cluster original data, cluster imputed data, calculate ARI
# Assumes imputed datasets already exist from main pipeline
# =============================================

# Load required libraries
library(readr)
library(dplyr)
library(devtools)
library(tidyr)
library(arrow) # For Feather support

source("Main.Pipe.r") # This is for the run_imputation_pipeline function

# Install and load IBclust if not already installed
if (!require(IBclust, quietly = TRUE)) {
  cat("Installing IBclust package from GitHub...\n")
  devtools::install_github("amarkos/IBclust")
  library(IBclust)
}

# Load cluster evaluation library for ARI calculation
if (!require(mclust, quietly = TRUE)) {
  install.packages("mclust")
  library(mclust)
}

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
  
enforce_original_types <- function(data, reference) {
    #' Enforce original data types from reference dataset
  #' 
  #' @param data Data to modify
  #' @param reference Reference dataset with correct types
  #' @return Data with corrected types
  for (col_name in names(data)) {
    if (col_name %in% names(reference)) {
      if (is.factor(reference[[col_name]])) {
        data[[col_name]] <- factor(data[[col_name]], levels = levels(reference[[col_name]]))
      } else if (is.numeric(reference[[col_name]])) {
        data[[col_name]] <- as.numeric(data[[col_name]])
      }
    }
  }
  return(data)
}

# Use embedded types for column identification
identify_column_types_embedded <- function(data) {
  cat_cols <- names(data)[sapply(data, function(x) is.factor(x) || is.character(x))]
  cont_cols <- names(data)[sapply(data, is.numeric)]
  return(list(
    categorical = cat_cols,
    continuous = cont_cols
  ))
}

perform_dibmix_clustering <- function(data, n_clusters = 2) {
  tryCatch({
    data <- as.data.frame(data)
    types <- identify_column_types_embedded(data)
    cat_cols <- types$categorical
    cont_cols <- types$continuous

    # Clean data for DIBmix
    if (length(cont_cols) > 0) {
      for (col in cont_cols) {
        data[[col]][is.infinite(data[[col]])] <- NA
      }
      data <- data[complete.cases(data[, cont_cols, drop = FALSE]), ]
    }
    if (length(cat_cols) > 0) {
      for (col in cat_cols) {
        if (is.factor(data[[col]])) {
          data[[col]] <- droplevels(data[[col]])
        }
        if (any(is.na(data[[col]]))) {
          data <- data[!is.na(data[[col]]), ]
        }
      }
    }

    # Perform clustering based on data types
    if (length(cat_cols) > 0 && length(cont_cols) > 0) {
      result <- DIBmix(X = data,
                       ncl = n_clusters,
                       catcols = which(names(data) %in% cat_cols),
                       contcols = which(names(data) %in% cont_cols),
                       s = -1,
                       lambda = -1,
                       nstart = 50)
    } else if (length(cont_cols) > 0) {
      X_cont <- as.matrix(data[, cont_cols, drop = FALSE])
      result <- DIBcont(X = X_cont, ncl = n_clusters, s = -1, nstart = 50)
    } else if (length(cat_cols) > 0) {
      X_cat <- data[, cat_cols, drop = FALSE]
      result <- DIBcat(X = X_cat, ncl = n_clusters, lambda = -1, nstart = 50)
    } else {
      stop("No valid columns found for clustering")
    }

    if (is.null(result) || !"Cluster" %in% names(result)) {
      stop("Clustering returned invalid result object")
    }

    result$categorical_cols <- cat_cols
    result$continuous_cols <- cont_cols
    result$n_clusters <- n_clusters

    cat(sprintf("Clustering completed: Entropy = %.4f, Mutual Info = %.4f\n",
                result$Entropy, result$MutualInfo))
    cat("Cluster assignments:", head(result$Cluster), "\n")

    return(result)


  }, error = function(e) {
    cat("\n!!! CLUSTERING FAILURE DETAILS !!!\n")
    cat("Error message:", e$message, "\n")
    cat("Data dimensions:", dim(data), "\n")
    return(NULL)
  })
}


calculate_ari <- function(true_clusters, pred_clusters) {
  if (length(true_clusters) != length(pred_clusters)) return(NA)
  valid_idx <- !is.na(true_clusters) & !is.na(pred_clusters)
  if (sum(valid_idx) == 0) return(NA)
  true_clean <- true_clusters[valid_idx]
  pred_clean <- pred_clusters[valid_idx]
  ari_value <- adjustedRandIndex(true_clean, pred_clean)
  return(ari_value)
}

# ----------------------------
# MAIN CLUSTERING EVALUATION FUNCTION
# ----------------------------

#################

evaluate_clustering_performance <- function(original_data_path,
                                           imputed_files_pattern = NULL,
                                           imputed_files_list = NULL,
                                           n_clusters = 3,
                                           output_file = "clustering_ari_results.feather") {
  cat("=== CLUSTERING EVALUATION ===\n")
  
  # Step 5: Load and cluster original dataset
  cat("\nStep 5: Clustering original dataset...\n")
  original_data <- arrow::read_feather(original_data_path)

  original_clustering <- perform_dibmix_clustering(
    original_data, 
    n_clusters
  )
  
  cat(sprintf("Original data dimensions: %d x %d\n", nrow(original_data), ncol(original_data)))
  
  if (is.null(original_clustering)) {
    cat("ERROR: Failed to cluster original data. Stopping.\n")
    return(NULL)
  }
  
  original_clusters <- original_clustering$Cluster
  cat(sprintf("Original clustering: %d clusters, Entropy = %.4f, Cluster vector length: %d\n", 
              n_clusters, original_clustering$Entropy, length(original_clusters)))
  
  # Get list of imputed files
  if (!is.null(imputed_files_pattern)) {
    imputed_files <- Sys.glob(imputed_files_pattern)
  } else if (!is.null(imputed_files_list)) {
    imputed_files <- imputed_files_list
  } else {
    imputed_files <- c()
    patterns <- c("*mice*.feather", "*famd*.feather", "*missforest*.feather")
    for (pattern in patterns) {
      imputed_files <- c(imputed_files, Sys.glob(pattern))
    }
  }
  
  if (length(imputed_files) == 0) {
    cat("ERROR: No imputed files found. Please specify imputed_files_pattern or imputed_files_list\n")
    return(NULL)
  }
  
  cat(sprintf("\nFound %d imputed files:\n", length(imputed_files)))
  for (f in imputed_files) cat(sprintf("  - %s\n", f))
  
  results <- data.frame()
  
  for (imputed_file in imputed_files) {
    method_name <- gsub(".*_([^_]+)_.*\\.feather$", "\\1", basename(imputed_file))
    if (method_name == basename(imputed_file)) {
      method_name <- gsub("\\.feather$", "", basename(imputed_file))
    }
    missing_rate_match <- regmatches(imputed_file, regexpr("0\\.[0-9]+", imputed_file))
    missing_rate <- if (length(missing_rate_match) > 0) as.numeric(missing_rate_match[1]) else NA
    
    tryCatch({
      imputed_data <- arrow::read_feather(imputed_file)
      imputed_data <- enforce_original_types(imputed_data, original_data)
      cat(sprintf("Imputed data dimensions: %d x %d\n", nrow(imputed_data), ncol(imputed_data)))
      
      if (nrow(imputed_data) != nrow(original_data)) {
        cat("WARNING: Row count mismatch (Original:", nrow(original_data), "vs Imputed:", nrow(imputed_data), "). Aligning using common indices.\n")
        imputed_data$.row_index <- 1:nrow(imputed_data)
        full_index <- data.frame(.row_index = 1:nrow(original_data))
        aligned_data <- full_index %>%
          left_join(imputed_data, by = ".row_index")
        aligned_data$.row_index <- NULL
        imputed_data <- aligned_data
        cat(sprintf("Aligned imputed data dimensions: %d x %d\n", nrow(imputed_data), ncol(imputed_data)))
      }      
      
      start_time <- Sys.time()
      imputed_clustering <- perform_dibmix_clustering(imputed_data, n_clusters)
      clustering_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      
      if (is.null(imputed_clustering)) {
        cat("WARNING: Clustering failed for this imputed dataset\n")
        ari_score <- NA
        imputed_entropy <- NA
        imputed_mutinfo <- NA
      } else {
        imputed_clusters <- imputed_clustering$Cluster
        imputed_entropy <- imputed_clustering$Entropy
        imputed_mutinfo <- imputed_clustering$MutualInfo
        if (length(original_clusters) != length(imputed_clusters)) {
          cat(sprintf("ERROR: Cluster vector length mismatch (original: %d, imputed: %d). Using intersection.\n",
                      length(original_clusters), length(imputed_clusters)))
          common_count <- min(length(original_clusters), length(imputed_clusters))
          orig_subset <- original_clusters[1:common_count]
          imp_subset <- imputed_clusters[1:common_count]
        } else {
          orig_subset <- original_clusters
          imp_subset <- imputed_clusters
        }
        ari_score <- calculate_ari(orig_subset, imp_subset)
        cat(sprintf("Imputed clustering: Entropy = %.4f, Mutual Info = %.4f\n",
                    imputed_entropy, imputed_mutinfo))
        cat(sprintf("ARI Score: %.4f\n", ari_score))
      }

      results <- rbind(results, data.frame(
        File = basename(imputed_file),
        Method = method_name,
        MissingRate = missing_rate,
        ARI = ari_score,
        OriginalEntropy = original_clustering$Entropy,
        OriginalMutInfo = original_clustering$MutualInfo,
        ImputedEntropy = imputed_entropy,
        ImputedMutInfo = imputed_mutinfo,
        ClusteringTimeSec = clustering_time,
        NClusters = n_clusters,
        NRows = nrow(imputed_data),
        NCols = ncol(imputed_data),
        stringsAsFactors = FALSE
      ))
      
    }, error = function(e) {
      cat("ERROR processing", imputed_file, ":", e$message, "\n")
      results <<- rbind(results, data.frame(
        File = basename(imputed_file),
        Method = method_name,
        MissingRate = missing_rate,
        ARI = NA,
        OriginalEntropy = original_clustering$Entropy,
        OriginalMutInfo = original_clustering$MutualInfo,
        ImputedEntropy = NA,
        ImputedMutInfo = NA,
        ClusteringTimeSec = NA,
        NClusters = n_clusters,
        NRows = NA,
        NCols = NA,
        stringsAsFactors = FALSE
      ))
    })
  }
  
  # arrow::write_feather(results, output_file)
  # cat(sprintf("\n=== RESULTS SAVED TO: %s ===\n", output_file))

  cat(sprintf("\n=== CLUSTERING RESULTS COMPLETE FOR DATASET ===\n"))
  
  # Print summary
  cat("\n=== SUMMARY ===\n")
  valid_results <- results[!is.na(results$ARI), ]
  
  if (nrow(valid_results) > 0) {
    cat("ARI Scores by Method:\n")
    summary_stats <- valid_results %>%
      group_by(Method) %>%
      summarise(
        Count = n(),
        Mean_ARI = mean(ARI, na.rm = TRUE),
        SD_ARI = sd(ARI, na.rm = TRUE),
        Min_ARI = min(ARI, na.rm = TRUE),
        Max_ARI = max(ARI, na.rm = TRUE),
        .groups = 'drop'
      ) %>%
      arrange(desc(Mean_ARI))
    
    print(summary_stats)
    
    # Best performing method overall
    best_method <- summary_stats$Method[1]
    best_ari <- summary_stats$Mean_ARI[1]
    cat(sprintf("\nBest performing method: %s (Mean ARI: %.4f)\n", best_method, best_ari))
    
    # If missing rates are available, show by missing rate
    if (any(!is.na(valid_results$MissingRate))) {
      cat("\nARI Scores by Missing Rate:\n")
      rate_summary <- valid_results %>%
        filter(!is.na(MissingRate)) %>%
        group_by(MissingRate) %>%
        summarise(
          Count = n(),
          Mean_ARI = mean(ARI, na.rm = TRUE),
          Best_Method = Method[which.max(ARI)],
          Best_ARI = max(ARI, na.rm = TRUE),
          .groups = 'drop'
        ) %>%
        arrange(MissingRate)
      
      print(rate_summary)
    }
  } else {
    cat("No valid results found. Check your data and clustering parameters.\n")
  }
  
  return(results)
}

# ----------------------------
# CONVENIENCE WRAPPER FUNCTIONS
# ----------------------------

# For when I know exactly which files to evaluate
evaluate_specific_files <- function(original_data_path, 
                                   imputed_files, 
                                   n_clusters = 3) {
  return(evaluate_clustering_performance(
    original_data_path = original_data_path,
    imputed_files_list = imputed_files,
    n_clusters = n_clusters
  ))
}

# For when imputed files follow a pattern
evaluate_by_pattern <- function(original_data_path, 
                               pattern = "*imputed*.feather", 
                               n_clusters = 3) {
  return(evaluate_clustering_performance(
    original_data_path = original_data_path,
    imputed_files_pattern = pattern,
    n_clusters = n_clusters
  ))
}
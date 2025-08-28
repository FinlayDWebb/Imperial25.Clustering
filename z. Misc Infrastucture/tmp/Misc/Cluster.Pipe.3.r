# =============================================
# CLUSTERING-ONLY EVALUATION SCRIPT (FEATHER VERSION)
# =============================================
# Steps 5-7: Cluster original data, cluster imputed data, calculate ARI
# Uses Feather's native type preservation - NO AUTOMATIC TYPE CONVERSION
# =============================================

# Load required libraries
library(dplyr)
library(devtools)
library(tidyr)
library(arrow) # For Feather support
library(mclust) # For ARI calculation

# Install and load IBclust if not already installed
if (!require(IBclust, quietly = TRUE)) {
  cat("Installing IBclust package from GitHub...\n")
  devtools::install_github("amarkos/IBclust")
  library(IBclust)
}

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

get_column_types <- function(data) {
  #' Identify column types based on embedded Feather metadata
  #' 
  #' @param data Dataframe to analyze
  #' @return List with categorical and continuous column indices
  
  cat_cols <- which(sapply(data, function(x) is.factor(x) || is.character(x)))
  cont_cols <- which(sapply(data, is.numeric))
  
  cat("Column type identification from embedded metadata:\n")
  cat(sprintf("  Categorical columns (%d): %s\n", 
              length(cat_cols), paste(names(data)[cat_cols], collapse = ", ")))
  cat(sprintf("  Continuous columns (%d): %s\n", 
              length(cont_cols), paste(names(data)[cont_cols], collapse = ", ")))
  
  return(list(
    categorical = cat_cols,
    continuous = cont_cols
  ))
}

perform_dibmix_clustering <- function(data, n_clusters = 3) {
  #' Perform DIBmix clustering using embedded types
  #' 
  #' @param data Dataframe to cluster (with preserved Feather types)
  #' @param n_clusters Number of clusters
  #' @return DIBmix clustering results
  
  tryCatch({
    # Create working copy preserving types
    data_copy <- as.data.frame(data)
    
    # Get column types from embedded metadata
    col_info <- get_column_types(data_copy)
    cat_cols <- col_info$categorical
    cont_cols <- col_info$continuous
    
    # Verify types match DIBmix requirements
    if (length(cat_cols) > 0) {
      # Ensure categoricals are factors
      data_copy[, cat_cols] <- lapply(data_copy[, cat_cols, drop = FALSE], function(x) {
        if (is.character(x)) as.factor(x) else x
      })
    }
    
    if (length(cont_cols) > 0) {
      # Ensure continuous are numeric
      data_copy[, cont_cols] <- lapply(data_copy[, cont_cols, drop = FALSE], as.numeric)
    }
    
    # Perform DIBmix clustering
    if (length(cat_cols) > 0 && length(cont_cols) > 0) {
      # Mixed-type data
      result <- DIBmix(X = data_copy, 
                       ncl = n_clusters, 
                       catcols = cat_cols, 
                       contcols = cont_cols,
                       s = -1,
                       lambda = -1,
                       nstart = 50)
    } else if (length(cont_cols) > 0) {
      # Continuous only
      X_cont <- as.matrix(data_copy[, cont_cols, drop = FALSE])
      result <- DIBcont(X = X_cont, ncl = n_clusters, s = -1, nstart = 50)
    } else if (length(cat_cols) > 0) {
      # Categorical only
      X_cat <- data_copy[, cat_cols, drop = FALSE]
      result <- DIBcat(X = X_cat, ncl = n_clusters, lambda = -1, nstart = 50)
    } else {
      stop("No valid columns found for clustering")
    }
    
    # Validate clustering result
    if (is.null(result) || !"Cluster" %in% names(result)) {
      stop("Clustering returned invalid result object")
    }
    
    # Add metadata
    result$n_clusters <- n_clusters
    
    cat(sprintf("Clustering completed: Entropy = %.4f, Mutual Info = %.4f\n",
                result$Entropy, result$MutualInfo))
    
    return(result)
    
  }, error = function(e) {
    cat("\n!!! CLUSTERING FAILURE DETAILS !!!\n")
    cat("Error message:", e$message, "\n")
    cat("Data dimensions:", dim(data), "\n")
    cat("Column types:", sapply(data, class), "\n")
    cat("n_clusters:", n_clusters, "\n")
    return(NULL)
  })
}

calculate_ari <- function(true_clusters, pred_clusters) {
  #' Calculate Adjusted Rand Index
  #' 
  #' @param true_clusters Reference cluster assignments
  #' @param pred_clusters Predicted cluster assignments
  #' @return ARI value
  
  if (length(true_clusters) != length(pred_clusters)) {
    cat("Error: Cluster vectors have different lengths\n")
    return(NA)
  }
  
  # Remove any NA values
  valid_idx <- !is.na(true_clusters) & !is.na(pred_clusters)
  if (sum(valid_idx) == 0) {
    cat("Error: No valid cluster assignments found\n")
    return(NA)
  }
  
  true_clean <- true_clusters[valid_idx]
  pred_clean <- pred_clusters[valid_idx]
  
  # Calculate ARI using mclust package
  ari_value <- adjustedRandIndex(true_clean, pred_clean)
  return(ari_value)
}

# ----------------------------
# MAIN CLUSTERING EVALUATION FUNCTION
# ----------------------------

evaluate_clustering_performance <- function(original_data_path,
                                           imputed_files_pattern = NULL,
                                           imputed_files_list = NULL,
                                           n_clusters = 3,
                                           output_file = "clustering_ari_results.feather") {
  cat("=== CLUSTERING EVALUATION ===\n")
  
  # Step 5: Load and cluster original dataset
  cat("\nStep 5: Clustering original dataset...\n")
  original_data <- arrow::read_feather(original_data_path)
  cat(sprintf("Original data dimensions: %d x %d\n", nrow(original_data), ncol(original_data)))
  
  original_clustering <- perform_dibmix_clustering(original_data, n_clusters)
  
  if (is.null(original_clustering)) {
    cat("ERROR: Failed to cluster original data. Stopping.\n")
    return(NULL)
  }
  
  original_clusters <- original_clustering$Cluster
  cat(sprintf("Original clustering: %d clusters, Entropy = %.4f\n", 
              n_clusters, original_clustering$Entropy))
  
  # Get list of imputed files
  if (!is.null(imputed_files_pattern)) {
    imputed_files <- Sys.glob(imputed_files_pattern)
  } else if (!is.null(imputed_files_list)) {
    imputed_files <- imputed_files_list
  } else {
    imputed_files <- Sys.glob("*_imputed.feather")
  }
  
  if (length(imputed_files) == 0) {
    cat("ERROR: No imputed files found\n")
    return(NULL)
  }
  
  cat(sprintf("\nFound %d imputed files:\n", length(imputed_files)))
  print(imputed_files)
  
  # Initialize results
  results <- data.frame()
  
  # Process each imputed dataset
  for (imputed_file in imputed_files) {
    method_name <- gsub(".*_([^_]+)_.*\\.feather$", "\\1", basename(imputed_file))
    if (method_name == basename(imputed_file)) {
      method_name <- tools::file_path_sans_ext(basename(imputed_file))
    }
    
    missing_rate_match <- regmatches(imputed_file, regexpr("0\\.[0-9]+", imputed_file))
    missing_rate <- if (length(missing_rate_match) > 0) as.numeric(missing_rate_match[1]) else NA
    
    tryCatch({
      cat(sprintf("\nProcessing: %s\n", basename(imputed_file)))
      
      # Load imputed data with preserved types
      imputed_data <- arrow::read_feather(imputed_file)
      cat(sprintf("Imputed data dimensions: %d x %d\n", nrow(imputed_data), ncol(imputed_data)))
      
      # Verify dimensions match
      if (nrow(imputed_data) != nrow(original_data)) {
        warning_msg <- sprintf(
          "Row count mismatch! Original: %d, Imputed: %d. Aligning rows.",
          nrow(original_data), nrow(imputed_data)
        )
        cat("WARNING:", warning_msg, "\n")
        
        # Create alignment index
        imputed_data$.row_index <- 1:nrow(imputed_data)
        full_index <- data.frame(.row_index = 1:nrow(original_data))
        
        # Merge to align rows
        aligned_data <- full_index %>%
          left_join(imputed_data, by = ".row_index")
        
        aligned_data$.row_index <- NULL
        imputed_data <- aligned_data
        cat(sprintf("Aligned dimensions: %d x %d\n", nrow(imputed_data), ncol(imputed_data)))
      }
      
      # Cluster imputed data
      start_time <- Sys.time()
      imputed_clustering <- perform_dibmix_clustering(imputed_data, n_clusters)
      clustering_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      
      if (is.null(imputed_clustering)) {
        cat("WARNING: Clustering failed for this dataset\n")
        ari_score <- NA
        imputed_entropy <- NA
        imputed_mutinfo <- NA
      } else {
        imputed_clusters <- imputed_clustering$Cluster
        imputed_entropy <- imputed_clustering$Entropy
        imputed_mutinfo <- imputed_clustering$MutualInfo
        
        # Handle possible length mismatches
        common_count <- min(length(original_clusters), length(imputed_clusters))
        orig_subset <- original_clusters[1:common_count]
        imp_subset <- imputed_clusters[1:common_count]
        
        # Calculate ARI
        ari_score <- calculate_ari(orig_subset, imp_subset)
        cat(sprintf("ARI Score: %.4f\n", ari_score))
      }
      
      # Store results
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
      cat("ERROR processing", basename(imputed_file), ":", e$message, "\n")
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
  
  # Save results
  arrow::write_feather(results, output_file)
  cat(sprintf("\nResults saved to: %s\n", output_file))
  
  # Print summary
  if (nrow(results) > 0) {
    cat("\n=== SUMMARY ===\n")
    print(results %>% select(Method, MissingRate, ARI, ClusteringTimeSec))
    
    if (!all(is.na(results$ARI))) {
      best_method <- results %>% 
        filter(!is.na(ARI)) %>% 
        arrange(desc(ARI)) %>% 
        slice(1) %>% 
        pull(Method)
      best_ari <- max(results$ARI, na.rm = TRUE)
      cat(sprintf("\nBest method: %s (ARI: %.4f)\n", best_method, best_ari))
    }
  }
  
  return(results)
}

# ----------------------------
# CONVENIENCE WRAPPER FUNCTIONS
# ----------------------------

evaluate_specific_files <- function(original_data_path, 
                                   imputed_files, 
                                   n_clusters = 3) {
  return(evaluate_clustering_performance(
    original_data_path = original_data_path,
    imputed_files_list = imputed_files,
    n_clusters = n_clusters
  ))
}

evaluate_by_pattern <- function(original_data_path, 
                               pattern = "*_imputed.feather", 
                               n_clusters = 3) {
  return(evaluate_clustering_performance(
    original_data_path = original_data_path,
    imputed_files_pattern = pattern,
    n_clusters = n_clusters
  ))
}
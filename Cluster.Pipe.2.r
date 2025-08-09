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

identify_column_types <- function(data) {
  #' Automatically identify categorical and continuous columns for DIBmix
  #' 
  #' @param data Dataframe to analyze
  #' @return List with categorical and continuous column indices
  
  cat_cols <- c()
  cont_cols <- c()
  
  for (i in 1:ncol(data)) {
    col <- data[[i]]
    col_name <- names(data)[i]
    
    if (is.factor(col) || is.character(col)) {
      cat_cols <- c(cat_cols, i)
      cat(sprintf("Column %d (%s): Categorical (%d levels)\n", 
                  i, col_name, length(unique(col[!is.na(col)]))))
    } else if (is.numeric(col)) {
      # Check if it's discrete (likely categorical) or continuous
      unique_vals <- length(unique(col[!is.na(col)]))
      total_vals <- sum(!is.na(col))
      
      if (unique_vals <= 10 || (total_vals > 0 && unique_vals / total_vals < 0.05)) {
        # Convert to factor and treat as categorical
        data[[i]] <- as.factor(col)
        cat_cols <- c(cat_cols, i)
        cat(sprintf("Column %d (%s): Converted to categorical (%d unique values)\n", 
                    i, col_name, unique_vals))
      } else {
        cont_cols <- c(cont_cols, i)
        cat(sprintf("Column %d (%s): Continuous (%d unique values)\n", 
                    i, col_name, unique_vals))
      }
    }
  }
  
  return(list(
    data = data,
    categorical = cat_cols, 
    continuous = cont_cols
  ))
}

perform_dibmix_clustering <- function(data, n_clusters = 3) {
  #' Perform DIBmix clustering on mixed-type data
  #' 
  #' @param data Dataframe to cluster
  #' @param n_clusters Number of clusters
  #' @return DIBmix clustering results

  tryCatch({
    data <- as.data.frame(data)  # Ensure data is a dataframe

    # Identify column types
    # col_info <- identify_column_types(data)
    # data_processed <- col_info$data
    # cat_cols <- col_info$categorical
    # cont_cols <- col_info$continuous

        # Identify column types (new)
    types <- identify_column_types_embedded(data)
    cat_cols <- types$categorical
    cont_cols <- types$continuous
    
    # Handle continuous variables: ensure numeric matrix
    if (length(cont_cols) > 0) {
      X_cont <- as.matrix(data[, cont_cols, drop = FALSE])
      mode(X_cont) <- "numeric"  # Force numeric type
    }
    
    # Handle categorical variables: ensure factor data frame
    if (length(cat_cols) > 0) {
      X_cat <- data[, cat_cols, drop = FALSE]
      # Convert to factors
      X_cat[] <- lapply(X_cat, function(x) if(!is.factor(x)) factor(x) else x)
    }
    
    # Perform clustering
    if (length(cat_cols) > 0 && length(cont_cols) > 0) {
      result <- DIBmix(
        X = X_cat,  # Categorical as factor data frame
        Y = X_cont,  # Continuous as numeric matrix
        ncl = n_clusters,
        s = -1,
        lambda = -1,
        nstart = 50
      )
    
    # --- CRITICAL DATA CONVERSION ---
    # Ensure continuous columns are numeric 
    } else if (length(cont_cols) > 0) {
      data_processed[, cont_cols] <- lapply(data_processed[, cont_cols, drop = FALSE], as.numeric)
    }
    
    # Ensure categorical columns are factors
    if (length(cat_cols) > 0) {
      data_processed[, cat_cols] <- lapply(data_processed[, cat_cols, drop = FALSE], function(x) {
        if(!is.factor(x)) as.factor(x) else x
      })
    }
    
    cat(sprintf("DIBmix clustering: %d categorical, %d continuous columns\n", 
                length(cat_cols), length(cont_cols)))
    cat("Data types after conversion:\n")
    print(sapply(data_processed, class))

    data_processed <- as.data.frame(data_processed)
    
    # Perform DIBmix clustering
    if (length(cat_cols) > 0 && length(cont_cols) > 0) {
      # Mixed-type data - use DIBmix
      result <- DIBmix(X = data_processed, 
                       ncl = n_clusters, 
                       catcols = cat_cols, 
                       contcols = cont_cols,
                       s = -1,
                       lambda = -1,
                       nstart = 50)
    } else if (length(cont_cols) > 0) {
      # Continuous only
      cat("Using DIBcont for continuous data\n")
      X_cont <- as.matrix(data_processed[, cont_cols, drop = FALSE])
      result <- DIBcont(X = X_cont, ncl = n_clusters, s = -1, nstart = 50)
    } else if (length(cat_cols) > 0) {
      # Categorical only
      cat("Using DIBcat for categorical data\n")
      X_cat <- data_processed[, cat_cols, drop = FALSE]
      result <- DIBcat(X = X_cat, ncl = n_clusters, lambda = -1, nstart = 50)
    } else {
      stop("No valid columns found for clustering")
    }
    
    # --- ERROR CHECK BEFORE RETURNING ---
    if (is.null(result) || !"Cluster" %in% names(result)) {
      stop("Clustering returned invalid result object")
    }
    
    # Add metadata
    result$column_info <- col_info
    result$n_clusters <- n_clusters
    
    cat(sprintf("Clustering completed: Entropy = %.4f, Mutual Info = %.4f\n",
                result$Entropy, result$MutualInfo))
    cat("Cluster assignments:", head(result$Cluster), "\n")
    
    return(result)
    
  }, error = function(e) {
    # --- ENHANCED ERROR REPORTING ---
    cat("\n!!! CLUSTERING FAILURE DETAILS !!!\n")
    cat("Error message:", e$message, "\n")
    cat("Data dimensions:", dim(data), "\n")
    if (exists("col_info")) {
      cat("Categorical columns:", length(col_info$categorical), "\n")
      cat("Continuous columns:", length(col_info$continuous), "\n")
    }
    cat("n_clusters:", n_clusters, "\n")
    return(NULL)
  })
}

calculate_ari <- function(true_clusters, pred_clusters) {
  #' Calculate Adjusted Rand Index between two clustering solutions
  #' 
  #' @param true_clusters Reference cluster assignments (from original data)
  #' @param pred_clusters Predicted cluster assignments (from imputed data)  
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

#################

evaluate_clustering_performance <- function(original_data_path,
                                           imputed_files_pattern = NULL,
                                           imputed_files_list = NULL,
                                           n_clusters = 3,
                                           output_file = "clustering_ari_results.csv") {
  cat("=== CLUSTERING EVALUATION ===\n")
  
  # Step 5: Load and cluster original dataset
  cat("\nStep 5: Clustering original dataset...\n")
  original_data <- read_csv(original_data_path, show_col_types = FALSE)
  cat(sprintf("Original data dimensions: %d x %d\n", nrow(original_data), ncol(original_data)))
  
  original_clustering <- perform_dibmix_clustering(original_data, n_clusters)
  
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
    # Default pattern - look for common imputed file names
    imputed_files <- c()
    patterns <- c("*mice*.csv", "*famd*.csv", "*missforest*.csv", "*midas*.csv")
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
  
# Initialize results
  results <- data.frame()
  
  # Step 6 & 7: Process each imputed dataset
  for (imputed_file in imputed_files) {
    cat(sprintf("\nProcessing: %s\n", basename(imputed_file)))
    
    # Extract method name from filename
    method_name <- gsub(".*_([^_]+)_.*\\.csv$", "\\1", basename(imputed_file))
    if (method_name == basename(imputed_file)) {
      # Fallback: use filename without extension
      method_name <- gsub("\\.csv$", "", basename(imputed_file))
    }
    
    # Extract missing rate if present in filename
    missing_rate_match <- regmatches(imputed_file, regexpr("0\\.[0-9]+", imputed_file))
    missing_rate <- if (length(missing_rate_match) > 0) as.numeric(missing_rate_match[1]) else NA
    
  tryCatch({
      # Step 6: Load imputed dataset
      imputed_data <- read_csv(imputed_file, show_col_types = FALSE)
      cat(sprintf("Imputed data dimensions: %d x %d\n", nrow(imputed_data), ncol(imputed_data)))
      
      # Ensure row count matches original
      if (nrow(imputed_data) != nrow(original_data)) {
        cat("WARNING: Row count mismatch (Original:", nrow(original_data), "vs Imputed:", nrow(imputed_data), "). Aligning using common indices.\n")
        
        # Add index column to preserve row order
        imputed_data$.row_index <- 1:nrow(imputed_data)
        
        # Create full index template
        full_index <- data.frame(.row_index = 1:nrow(original_data))
        
        # Merge to align rows
        aligned_data <- full_index %>%
          left_join(imputed_data, by = ".row_index")
        
        # Remove index column
        aligned_data$.row_index <- NULL
        
        imputed_data <- aligned_data
        cat(sprintf("Aligned imputed data dimensions: %d x %d\n", nrow(imputed_data), ncol(imputed_data)))
      }      
      
      # Cluster imputed dataset
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
        
        # CRITICAL FIX: Validate cluster vector length
        if (length(original_clusters) != length(imputed_clusters)) {
          cat(sprintf("ERROR: Cluster vector length mismatch (original: %d, imputed: %d). Using intersection.\n",
                      length(original_clusters), length(imputed_clusters)))
          
          # Use only common indices
          common_count <- min(length(original_clusters), length(imputed_clusters))
          orig_subset <- original_clusters[1:common_count]
          imp_subset <- imputed_clusters[1:common_count]
        } else {
          orig_subset <- original_clusters
          imp_subset <- imputed_clusters
        }
        
        # Step 7: Calculate ARI
        ari_score <- calculate_ari(orig_subset, imp_subset)
        
        cat(sprintf("Imputed clustering: Entropy = %.4f, Mutual Info = %.4f\n",
                    imputed_entropy, imputed_mutinfo))
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
      cat("ERROR processing", imputed_file, ":", e$message, "\n")
      
      # Store error result
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
  write_csv(results, output_file)
  cat(sprintf("\n=== RESULTS SAVED TO: %s ===\n", output_file))
  
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
                               pattern = "*imputed*.csv", 
                               n_clusters = 3) {
  return(evaluate_clustering_performance(
    original_data_path = original_data_path,
    imputed_files_pattern = pattern,
    n_clusters = n_clusters
  ))
}

# =============================================
# EXAMPLE USAGE
# =============================================

# Example 1: Evaluate specific imputed files
if (FALSE) {  # Set to TRUE to run
  results <- evaluate_specific_files(
    original_data_path = "input_data",
    imputed_files = c(
      "mice_0.05_imputed.csv",
      "famd_0.05_imputed.csv", 
      "missforest_0.05_imputed.csv",
      "midas_0.05_pooled.csv",
      "mice_0.10_imputed.csv",
      "famd_0.10_imputed.csv",
      "missforest_0.10_imputed.csv",
        "midas_0.10_pooled.csv",
        "mice_0.15_imputed.csv",
        "famd_0.15_imputed.csv",
        "missforest_0.15_imputed.csv",
        "midas_0.15_pooled.csv"
    ),
    n_clusters = 3
  )
}

# Example 2: Evaluate all CSV files with "imputed" in the name
if (FALSE) {  # Set to TRUE to run
  results <- evaluate_by_pattern(
    original_data_path = "adult_sample_processed.csv",
    pattern = "*imputed*.csv",
    n_clusters = 3
  )
}

# Example 3: Manual specification with custom parameters
if (FALSE) {  # Set to TRUE to run
  results <- evaluate_clustering_performance(
    original_data_path = "adult_sample_processed.csv",
    imputed_files_pattern = "*_pooled.csv",  # For MIDAS files
    n_clusters = 4,  # Different number of clusters
    output_file = "my_clustering_results.csv"
  )
  
  print(head(results))
}
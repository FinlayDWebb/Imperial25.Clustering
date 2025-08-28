# =============================================
# MISSING DATA IMPUTATION PIPELINE
# =============================================
# Preprocessing -> MAR Insertion -> Imputation -> Evaluation
# =============================================

# Load required libraries
library(readr)
library(dplyr)
library(mice)
library(missMDA)
library(missForest)
library(tidyr)
library(arrow)

# ----------------------------
# 1. DATA PREPROCESSING
# ----------------------------

preprocess_data <- function(file_path) {
  # Read Feather file and use embedded types
  data <- arrow::read_feather(file_path)
  # Optionally trim whitespace from character columns
  data <- data %>% mutate(across(where(is.character), trimws))
  # No conversion of character to factor here; keep native types
  return(as.data.frame(data))
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

# ----------------------------
# 2. MAR MISSINGNESS INSERTION
# ----------------------------

insert_mar <- function(data, target_cols, predictor_cols, missing_rate) {
  #' Inserts MAR missingness using logistic regression principles
  #'
  #' @param data Complete dataframe
  #' @param target_cols Columns to introduce missingness
  #' @param predictor_cols Columns influencing missingness
  #' @param missing_rate Target proportion of missing values (0-1)
  #' @return Dataframe with MAR missingness
  
  set.seed(42)
  data_mar <- data
  
  for (target_col in target_cols) {
    # Select random predictor if not specified
    if (is.null(predictor_cols)) {
      available_preds <- setdiff(names(data), target_col)
      predictor <- sample(available_preds, 1)
    } else {
      available_preds <- setdiff(predictor_cols, target_col)
      if (length(available_preds) == 0) available_preds <- sample(setdiff(names(data), target_col), 1)
      predictor <- sample(available_preds, 1)
    }
    
    # Convert predictor to numeric representation
    if (is.numeric(data_mar[[predictor]])) {
      pred_vals <- data_mar[[predictor]]
      # Check if column has variance before scaling
      if (var(pred_vals, na.rm = TRUE) > 0) {
        pred_vals <- as.numeric(scale(pred_vals))
      } else {
        # Handle constant columns - use small random variation
        pred_vals <- rep(0, length(pred_vals))
      }
    } else {
      # Convert factors/characters to numeric embeddings
      pred_vals <- as.numeric(factor(data_mar[[predictor]]))
      # Check if column has variance before scaling
      if (var(pred_vals, na.rm = TRUE) > 0) {
        pred_vals <- as.numeric(scale(pred_vals))
      } else {
        # Handle constant columns - use small random variation
        pred_vals <- rep(0, length(pred_vals))
      }
    }
    
    # Ensure no NaN or NA values in pred_vals
    pred_vals[is.na(pred_vals) | is.nan(pred_vals)] <- 0
    
    # Create missingness mechanism: logit(p) = intercept + beta * predictor
    beta <- 2  # Fixed effect size
    log_odds <- beta * pred_vals
    p_missing <- 1 / (1 + exp(-log_odds))
    
    # Handle any remaining NaN/NA values in probabilities
    p_missing[is.na(p_missing) | is.nan(p_missing)] <- 0.5
    
    # Adjust probabilities to hit target missing rate
    current_mean <- mean(p_missing, na.rm = TRUE)
    if (current_mean > 0) {
      p_scale <- missing_rate / current_mean
      p_missing <- pmin(p_missing * p_scale, 0.95)  # Cap at 95% to avoid all-missing
    } else {
      # If all probabilities are 0, set uniform probability
      p_missing <- rep(missing_rate, length(p_missing))
    }
    
    # Introduce missing values
    missing_mask <- runif(nrow(data_mar)) < p_missing
    
    # Final safety check - ensure missing_mask has no NA values
    missing_mask[is.na(missing_mask)] <- FALSE
    
    data_mar[missing_mask, target_col] <- NA
  }
  
  actual_rate <- mean(sapply(target_cols, function(col) mean(is.na(data_mar[col]))))
  cat(sprintf("Introduced MAR missingness. Target: %.1f%%, Actual: %.1f%%\n",
              missing_rate*100, actual_rate*100))
  
  return(data_mar)
}

# ----------------------------
# 3. IMPUTATION METHODS
# ----------------------------

impute_mice <- function(data) {
  cat("\n=== MICE Imputation with Pooling ===\n")
  # Use factors as they are, do not force conversion of character columns
  set.seed(42)
  pred_matrix <- make.predictorMatrix(data)
  methods <- make.method(data)
  methods[sapply(data, is.factor)] <- "polyreg"
  methods[sapply(data, is.numeric)] <- "pmm"
  cat("Imputation methods:\n")
  print(methods[methods != ""])
  imp <- mice(data,
             m = 5,
             maxit = 10,
             method = methods,
             predictorMatrix = pred_matrix,
             printFlag = TRUE,
             seed = 42)
  completed_datasets <- lapply(1:5, function(i) complete(imp, i))
  pooled_data <- data.frame(matrix(nrow = nrow(data), ncol = ncol(data)))
  names(pooled_data) <- names(data)
  for (col in names(data)) {
    if (is.numeric(data[[col]])) {
      values_matrix <- sapply(completed_datasets, function(df) df[[col]])
      pooled_data[[col]] <- rowMeans(values_matrix, na.rm = TRUE)
    } else if (is.factor(data[[col]])) {
      pooled_values <- character(nrow(data))
      for (row in 1:nrow(data)) {
        imputed_values <- sapply(completed_datasets, function(df) as.character(df[[col]][row]))
        mode_value <- names(sort(table(imputed_values), decreasing = TRUE))[1]
        pooled_values[row] <- mode_value
      }
      pooled_data[[col]] <- factor(pooled_values, levels = levels(data[[col]]))
    } else {
      pooled_data[[col]] <- data[[col]]
    }
  }
  for (col in names(data)) {
    if (is.factor(data[[col]])) {
      pooled_data[[col]] <- factor(pooled_data[[col]], levels = levels(data[[col]]))
    }
  }
  cat("Pooling complete. Missing values after pooling:", sum(is.na(pooled_data)), "\n")
  return(pooled_data)
}

impute_famd <- function(data) {
  set.seed(42)
  
  # Identify categorical and continuous columns from Feather types
  types <- identify_column_types_embedded(data)
  cat_cols <- types$categorical
  cont_cols <- types$continuous

  # Convert categorical columns to factors if not already
  if(length(cat_cols) > 0) {
    data[cat_cols] <- lapply(data[cat_cols], function(x) if(!is.factor(x)) factor(x) else x)
  }

  # Store original factor levels
  factor_levels <- lapply(data[cat_cols], levels)

  # Estimate optimal ncp
  message("Estimating optimal number of components for FAMD...")
  ncp_est <- missMDA::estim_ncpFAMD(data)$ncp
  message("Optimal ncp estimated: ", ncp_est)

  tryCatch({
    # Impute
    imp <- missMDA::imputeFAMD(data, ncp = ncp_est, method = "Regularized")$completeObs

    # Restore original factor levels
    for(col in cat_cols) {
      imp[[col]] <- factor(imp[[col]], levels = factor_levels[[col]])
    }

    return(imp)
  }, error = function(e) {
    message("FAMD failed: ", e$message)
    return(NULL)
  })
}


impute_missforest <- function(data) {
  #' Random forest-based imputation with simplified, robust handling
  #' 
  #' @param data Dataframe with missing values
  #' @return Imputed dataframe
  
  set.seed(42)
  
  # Store original factor levels for restoration
  factor_levels <- lapply(data, function(x) if(is.factor(x)) levels(x) else NULL)
  
  tryCatch({
    # Convert to data frame (missForest requires data frame)
    data_df <- as.data.frame(data)
    
    # Simple, clean data preparation - no aggressive conversions
    # Just ensure characters are factors (they should already be from preprocessing)
    data_df <- data_df %>% mutate(across(where(is.character), as.factor))
    
    # Basic validation
    if (nrow(data_df) == 0 || ncol(data_df) == 0) {
      cat("Error: Empty dataset\n")
      return(NULL)
    }
    
    # Check for entirely missing columns
    entirely_missing_cols <- sapply(data_df, function(x) all(is.na(x)))
    if (any(entirely_missing_cols)) {
      cat("Warning: Removing entirely missing columns:", names(data_df)[entirely_missing_cols], "\n")
      data_df <- data_df[, !entirely_missing_cols, drop = FALSE]
    }
    
    # Check for entirely missing rows
    entirely_missing_rows <- apply(data_df, 1, function(x) all(is.na(x)))
    if (any(entirely_missing_rows)) {
      cat("Warning: Removing entirely missing rows:", sum(entirely_missing_rows), "\n")
      data_df <- data_df[!entirely_missing_rows, , drop = FALSE]
    }
    
    # Show data types for debugging
    cat("Data types going into missForest:\n")
    for (col in names(data_df)) {
      cat(sprintf("  %s: %s", col, class(data_df[[col]])[1]))
      if (is.factor(data_df[[col]])) {
        cat(sprintf(" (%d levels)", nlevels(data_df[[col]])))
      }
      cat("\n")
    }
    
    # Perform missForest with conservative settings
    cat("Performing missForest...\n")
    forest_result <- missForest::missForest(
      xmis = data_df,
      maxiter = 10,
      ntree = 100,
      verbose = FALSE
    )
    
    # Get imputed data
    imputed_data <- forest_result$ximp
    
    # Restore original factor levels if needed
    for (col in names(data)) {
      if (is.factor(data[[col]]) && !is.null(factor_levels[[col]])) {
        if (col %in% names(imputed_data)) {
          imputed_data[[col]] <- factor(imputed_data[[col]], levels = factor_levels[[col]])
        }
      }
    }
    
    # Handle dimension mismatch if rows were removed
    if (nrow(imputed_data) != nrow(data)) {
      cat("Warning: Row count mismatch. Creating result with original dimensions.\n")
      result <- data
      if (exists("entirely_missing_rows")) {
        result[!entirely_missing_rows, names(imputed_data)] <- imputed_data
      }
      return(result)
    }
    
    return(imputed_data)
    
  }, error = function(e) {
    cat("missForest imputation failed with error:", e$message, "\n")
    
    # Detailed diagnostics
    cat("Detailed diagnostics:\n")
    cat("- Data class:", class(data), "\n")
    cat("- Data dimensions:", dim(data), "\n")
    
    # Check each column
    for (col in names(data)) {
      col_data <- data[[col]]
      cat(sprintf("- %s: class=%s", col, paste(class(col_data), collapse=",")))
      
      if (is.factor(col_data)) {
        cat(sprintf(", levels=%d, na_count=%d", nlevels(col_data), sum(is.na(col_data))))
        
        # Check for problematic factor levels
        levels_char <- levels(col_data)
        if (any(is.na(levels_char)) || any(levels_char == "")) {
          cat(" [PROBLEMATIC LEVELS DETECTED]")
        }
      } else if (is.numeric(col_data)) {
        cat(sprintf(", na_count=%d, range=[%.2f,%.2f]", 
                   sum(is.na(col_data)), 
                   min(col_data, na.rm=TRUE), 
                   max(col_data, na.rm=TRUE)))
      }
      cat("\n")
    }
    
    return(NULL)
  })
}

# ----------------------------
# EVALUATION METRICS FUNCTIONS
# ----------------------------

calculate_rmse <- function(original, imputed) {
  #' Calculates RMSE for numerical columns
  #' 
  #' @param original Complete original dataframe
  #' @param imputed Imputed dataframe
  #' @return Mean RMSE across numerical columns
  
  num_cols <- names(original)[sapply(original, is.numeric)]
  
  if (length(num_cols) == 0) return(NA)
  
  rmse_vals <- sapply(num_cols, function(col) {
    sqrt(mean((original[[col]] - imputed[[col]])^2, na.rm = TRUE))
  })
  
  mean(rmse_vals, na.rm = TRUE)
}

calculate_pfc <- function(original, imputed) {
  #' Calculates PFC for categorical columns
  #' 
  #' @param original Complete original dataframe
  #' @param imputed Imputed dataframe
  #' @return Mean PFC across categorical columns
  
    cat_cols <- names(original)[sapply(original, function(x) {
    is.factor(x) || (is.numeric(x) && length(unique(x)) <= 10)
  })]
  
  if (length(cat_cols) == 0) return(NA)
  
  pfc_vals <- sapply(cat_cols, function(col) {
    orig_char <- as.character(original[[col]])
    imp_char <- as.character(imputed[[col]])
    mean(orig_char != imp_char, na.rm = TRUE)
  })
  
  mean(pfc_vals, na.rm = TRUE)
}

# ----------------------------
# 6. MAIN PIPELINE
# ----------------------------

run_imputation_pipeline <- function(data_path, 
                                    missing_rates = c(0.05, 0.10, 0.15),
                                    methods = c("MICE", "FAMD", "missForest")) {

  # Extract dataset name for file prefixes
  base_name <- tools::file_path_sans_ext(basename(data_path))

  # 1. Preprocess data
  clean_data <- preprocess_data(data_path)
  
  # Initialize results
  results <- data.frame()
  
  for (rate in missing_rates) {
    cat("\n=== Processing Missing Rate:", paste0(rate*100, "%"), "===\n")
    
    # 2. Introduce MAR missingness
    mar_data <- insert_mar(clean_data, 
                           target_cols = names(clean_data), # All columns
                           predictor_cols = NULL,         # Random predictors
                           missing_rate = rate)
    
    for (method in methods) {
      cat("\nTrying method:", method, "\n")
      
      # 3. Impute missing values
      start_time <- Sys.time()
      imputed_data <- NULL
      
      imputed_data <- tryCatch({
        switch(method,
               "MICE" = impute_mice(mar_data),
               "FAMD" = impute_famd(mar_data),
               "missForest" = impute_missforest(mar_data)
        )
      }, error = function(e) {
        cat(method, "imputation failed:", e$message, "\n")
        NULL
      })
    
      if (is.null(imputed_data)) {
        cat(method, "imputation returned NULL, skipping...\n")
        next
      }

      # Save the imputed dataset to Feather
      output_filename <- file.path("imputed_datasets", 
                            sprintf("%s_%s_%.2f_imputed.feather", 
                                    base_name, tolower(method), rate))
      write_feather(imputed_data, output_filename)
      cat("Saved imputed data to:", output_filename, "\n")
      
      time_taken <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      
      # 4. Calculate evaluation metrics
      rmse <- calculate_rmse(clean_data, imputed_data)
      pfc <- calculate_pfc(clean_data, imputed_data)
      
      # 5. Store results
      results <- rbind(results, data.frame(
        MissingRate = rate,
        Method = method,
        RMSE = rmse,
        PFC = pfc,
        TimeSec = time_taken,
        NumRows = nrow(clean_data),
        NumCols = ncol(clean_data)
      ))
      
      # Print progress
      cat(sprintf(
        "✓ Method: %-10s | Rate: %-4s | RMSE: %.4f | PFC: %.4f | Time: %.1fs\n",
        method, paste0(rate*100, "%"), rmse, pfc, time_taken
      ))
    }
  }
  
  # Save results (keeping as CSV for compatibility)
  write_csv(results, "imputation_results.csv")
  cat("\n=== PIPELINE COMPLETE ===\n")
  cat("Results saved to 'imputation_results.csv'\n")
  
  return(results)
}
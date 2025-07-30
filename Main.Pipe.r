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

# ----------------------------
# 1. DATA PREPROCESSING
# ----------------------------

preprocess_data <- function(file_path) {
  #' Cleans and prepares raw data for analysis (more robust version)
  #'
  #' @param file_path Path to raw data file
  #' @return Cleaned dataframe with proper data types
  
  # Read data treating "?" as missing from the start
  data <- read_csv(file_path, 
                   na = c("", "NA", "?"),  # Handle "?" during reading
                   show_col_types = FALSE)
  
  # Basic cleaning operations
  clean_data <- data %>%
    # Remove leading/trailing whitespace from character columns
    mutate(across(where(is.character), trimws)) %>%
    # Convert character columns to factors
    mutate(across(where(is.character), as.factor))
  
  cat("Data preprocessing complete\n")
  cat("Dimensions:", dim(clean_data), "\n")
  
  return(clean_data)
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
  #' MICE imputation with multiple datasets and pooling
  #' 
  #' @param data Dataframe with missing values
  #' @return Pooled imputed dataframe
  
  cat("\n=== MICE Imputation with Pooling ===\n")
  
  # Ensure factors are properly set
  data <- data %>% mutate(across(where(is.character), as.factor))
  
  set.seed(42)
  
  # Create predictor matrix - exclude categorical variables from PMM for other categoricals
  pred_matrix <- make.predictorMatrix(data)
  
  # Set methods explicitly
  methods <- make.method(data)
  methods[sapply(data, is.factor)] <- "polyreg"  # Use polytomous regression for factors
  methods[sapply(data, is.numeric)] <- "pmm"     # Use PMM for numeric
  
  cat("Imputation methods:\n")
  print(methods[methods != ""])
  
  # Perform MICE with multiple imputations (m=5)
  imp <- mice(data,
             m = 5,                    # Create 5 imputed datasets
             maxit = 10,
             method = methods,
             predictorMatrix = pred_matrix,
             printFlag = TRUE,         # Show progress
             seed = 42)
  
  # Pool the results
  cat("\nPooling 5 imputations...\n")
  
  # Extract all completed datasets
  completed_datasets <- vector("list", 5)
  for (i in 1:5) {
    completed_datasets[[i]] <- complete(imp, i)
  }
  
  # Pool continuous variables using mean
  # Pool categorical variables using mode
  pooled_data <- data.frame(matrix(nrow = nrow(data), ncol = ncol(data)))
  names(pooled_data) <- names(data)
  
  for (col in names(data)) {
  if (is.numeric(data[[col]])) {
    # ADD NA CHECK AND IMPUTATION
    values_matrix <- sapply(completed_datasets, function(df) {
      col_vals <- df[[col]]
      if(any(is.na(col_vals))) {
        # Impute with mean if NAs exist
        col_vals[is.na(col_vals)] <- mean(col_vals, na.rm = TRUE)
      }
      col_vals
    })
    pooled_data[[col]] <- rowMeans(values_matrix)
    } else if (is.factor(data[[col]])) {
      # For categorical variables: use mode (most frequent) across imputations
      pooled_values <- character(nrow(data))
      for (row in 1:nrow(data)) {
        if (is.na(data[[col]][row])) {
          # Get values from all imputations for this missing cell
          imputed_values <- sapply(completed_datasets, function(df) as.character(df[[col]][row]))
          # Find mode (most frequent value)
          mode_value <- names(sort(table(imputed_values), decreasing = TRUE))[1]
          pooled_values[row] <- mode_value
        } else {
          # Keep original observed value
          pooled_values[row] <- as.character(data[[col]][row])
        }
      }
      pooled_data[[col]] <- factor(pooled_values, levels = levels(data[[col]]))
    }
  }
  
  # Ensure factor levels match original data
  for (col in names(data)) {
    if (is.factor(data[[col]])) {
      pooled_data[[col]] <- factor(pooled_data[[col]], levels = levels(data[[col]]))
    }
  }
  
  cat("Pooling complete. Missing values after pooling:", sum(is.na(pooled_data)), "\n")
  
  return(pooled_data)
}

# Perhaps this won't work with the high categorical datasets. I need to re-think this. Perhaps
# use some more choice datasets.

impute_famd <- function(data, ncp = 10) {
  set.seed(42)
  
  # 1. Convert only categorical columns to factors
  data <- as.data.frame(lapply(data, function(x) {
    if (is.character(x) || is.logical(x) || (is.numeric(x) && length(unique(x)) <= 10)) {
      factor(x)
    } else {
      x
    }
  }))
  
  # 2. Store original levels
  factor_levels <- lapply(data, function(x) if (is.factor(x)) levels(x) else NULL)
  
  tryCatch({
    # 3. Perform imputation
    imp <- missMDA::imputeFAMD(data, ncp = ncp)$completeObs
    
    # 4. Convert probability vectors back to categories
    for (col in names(data)) {
      if (is.factor(data[[col]])) {
        # Find probability columns (they start with colname + ".")
        prob_cols <- grep(paste0("^", col, "\\."), names(imp), value = TRUE)
        
        if (length(prob_cols) > 0) {
          # Get probability matrix
          probs <- as.matrix(imp[, prob_cols])
          
          # Assign category with highest probability
          max_cat <- apply(probs, 1, which.max)
          imp[[col]] <- factor_levels[[col]][max_cat]
        }
      }
    }
    
    # 5. CRITICAL FIX: Convert back to factors with original levels
    for (col in names(data)) {
      if (is.factor(data[[col]])) {
        imp[[col]] <- factor(imp[[col]], levels = factor_levels[[col]])
      }
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
# 5. MIDAS IMPUTATION HANDLING
# ----------------------------

process_midas_imputations <- function(file_pattern, output_file, num_files = 5) {
  #' Process multiple MIDAS imputation files and create pooled result
  
  # Read all MIDAS imputations
  imputations <- list()
  files_found <- 0
  
  for (i in 1:num_files) {
    file <- sprintf(file_pattern, i)
    if (file.exists(file)) {
      imputations[[length(imputations) + 1]] <- read_csv(file, show_col_types = FALSE)
      files_found <- files_found + 1
    }
  }

  if (files_found == 0) {
    cat("No MIDAS imputation files found with pattern:", file_pattern, "\n")
    return(NULL)
  }
  
  cat("Found", files_found, "MIDAS imputation files\n")
  
  # Create the averaged dataset
  pooled <- imputations[[1]]
  
  # Numeric columns: average across imputations
  num_cols <- names(pooled)[sapply(pooled, is.numeric)]
  for (col in num_cols) {
    values <- sapply(imputations, function(df) df[[col]])
    pooled[[col]] <- rowMeans(values, na.rm = TRUE)
  }
  
  # Categorical columns: use mode
  cat_cols <- names(pooled)[sapply(pooled, function(x) is.factor(x) || is.character(x))]
  for (col in cat_cols) {
    values <- sapply(imputations, function(df) as.character(df[[col]]))
    pooled[[col]] <- apply(values, 1, function(row) {
      ux <- unique(row[!is.na(row)])
      if (length(ux) == 0) return(NA)
      ux[which.max(tabulate(match(row, ux)))]
    })
    
    # Convert back to factor if original was factor
    if (is.factor(imputations[[1]][[col]])) {
      pooled[[col]] <- factor(pooled[[col]], levels = levels(imputations[[1]][[col]]))
    }
  }
  
  # Save pooled imputation
  write_csv(pooled, output_file)
  cat("Saved pooled MIDAS imputation to", output_file, "\n")
  
  return(pooled)
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
    # Handle the prefix that MIDASpy adds
    orig_char <- as.character(original[[col]])
    imp_char <- as.character(imputed[[col]])
    
    # Remove the column name prefix if present (e.g., "workclass_Private" -> "Private")
    imp_char <- gsub(paste0("^", col, "_"), "", imp_char)

    # Convert both to character to handle 1.0 vs 1 issue
    orig_char <- as.character(original[[col]])
    imp_char <- as.character(imputed[[col]])
    
    # For numeric-like factors, also try removing decimal zeros
    if (all(grepl("^\\d+(\\.0*)?$", imp_char, na.rm = TRUE))) {
      imp_char <- gsub("\\.0+$", "", imp_char)
    }
    
    mean(orig_char != imp_char, na.rm = TRUE)
  })
  
  mean(pfc_vals, na.rm = TRUE)
}

# ----------------------------
# 6. MAIN PIPELINE
# ----------------------------

run_imputation_pipeline <- function(data_path, 
                                    missing_rates = c(0.05, 0.10, 0.15),
                                    methods = c("MICE", "FAMD", "missForest", "MIDAS")) {

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
      
      if (method == "MIDAS") {
        # Special handling for MIDAS
        midas_input <- sprintf("%s_mar_%.2f.csv", base_name, rate)
        midas_output_prefix <- sprintf("%s_midas_%.2f", base_name, rate)
        
        # Save temporary file for Python processing
        write_csv(mar_data, midas_input)
        cat("Saved input file:", midas_input, "\n")
        
        # Call Python script (assumes it's in same directory)
        python_cmd <- sprintf("python MIDAS.Pipe.py %s %s", midas_input, midas_output_prefix)
        cat("Running:", python_cmd, "\n")
        system(python_cmd)

        # Process and pool MIDAS imputations
        imputed_data <- process_midas_imputations(
          file_pattern = paste0(midas_output_prefix, "_imp_%d.csv"),
          output_file = paste0(midas_output_prefix, "_pooled.csv")
        )
        
        # Clean up temporary file
        if (file.exists(midas_input)) file.remove(midas_input)
        
      } else {
        # R-based methods
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
      }
    
      if (is.null(imputed_data)) {
        cat(method, "imputation returned NULL, skipping...\n")
        next
      }

      # Save the imputed dataset to CSV ***
      output_filename <- sprintf("%s_%s_%.2f_imputed.csv", base_name, tolower(method), rate)
      write_csv(imputed_data, output_filename)
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
  
  # Save results
  write_csv(results, "imputation_results.csv")
  cat("\n=== PIPELINE COMPLETE ===\n")
  cat("Results saved to 'imputation_results.csv'\n")
  
  return(results)
}

# This is executed by Total.Pipe.r
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
  
  # Remove rows with any missing values
  complete_data <- clean_data %>% 
    drop_na() %>%
    distinct()
  
  cat("Data preprocessing complete\n")
  cat("Original dimensions:", dim(data), "\n")
  cat("Cleaned dimensions:", dim(complete_data), "\n")
  
  return(complete_data)
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
  #' MICE imputation using predictive mean matching
  #' 
  #' @param data Dataframe with missing values
  #' @return Imputed dataframe
  
  set.seed(42)
  imp <- mice(data, m = 1, maxit = 10, method = "pmm")
  complete(imp)
}

impute_famd <- function(data) {
  #' FAMD imputation using factor analysis
  #' 
  #' @param data Dataframe with missing values
  #' @return Imputed dataframe
  
  set.seed(42)
  imp <- imputeFAMD(data, ncp = 2)$completeObs
  imp %>% mutate(across(where(is.factor), as.factor))
}

impute_missforest <- function(data) {
  #' Random forest-based imputation
  #' 
  #' @param data Dataframe with missing values
  #' @return Imputed dataframe
  
  set.seed(42)
  missForest(data)$ximp
}

# ----------------------------
# 4. EVALUATION METRICS
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
  
  cat_cols <- names(original)[sapply(original, is.factor)]
  
  if (length(cat_cols) == 0) return(NA)
  
  pfc_vals <- sapply(cat_cols, function(col) {
    mean(original[[col]] != imputed[[col]], na.rm = TRUE)
  })
  
  mean(pfc_vals, na.rm = TRUE)
}

# ----------------------------
# 5. MIDAS IMPUTATION HANDLING
# ----------------------------

process_midas_imputations <- function(file_pattern, output_file, num_files = 5) {
  # Read all MIDAS imputations
  imputations <- list()
  for (i in 1:num_files) {
    file <- sprintf(file_pattern, i)
    if (file.exists(file)) {
      imputations[[i]] <- read_csv(file, show_col_types = FALSE)
    }
  }

  if (length(imputations) == 0) return(NULL)
  
  # Create the averaged dataset
  pooled <- imputations[[1]]
  
  # Numeric columns: average across imputations
  num_cols <- names(pooled)[sapply(pooled, is.numeric)]
  for (col in num_cols) {
    values <- sapply(imputations, function(df) df[[col]])
    pooled[[col]] <- rowMeans(values, na.rm = TRUE)
  }
  
   
  # Categorical columns: use mode
  cat_cols <- names(pooled)[sapply(pooled, is.factor)]
  for (col in cat_cols) {
    values <- sapply(imputations, function(df) as.character(df[[col]]))
    pooled[[col]] <- apply(values, 1, function(row) {
      ux <- unique(row)
      ux[which.max(tabulate(match(row, ux)))]
    }) %>% factor(levels = levels(pooled[[col]]))
  }
  
  # Save pooled imputation
  write_csv(pooled, output_file)
  cat("Saved pooled MIDAS imputation to", output_file, "\n")
  
  return(pooled)
}

# ----------------------------
# 6. MAIN PIPELINE
# ----------------------------

run_imputation_pipeline <- function(data_path, 
                                    missing_rates = c(0.05, 0.10, 0.15),
                                    methods = c("MICE", "FAMD", "missForest", "MIDAS")) {
  # 1. Preprocess data
  clean_data <- preprocess_data(data_path)
  
  # Initialize results
  results <- data.frame()
  
  for (rate in missing_rates) {
    # 2. Introduce MCAR missingness
    mar_data <- insert_mar(clean_data, 
                           target_cols = names(clean_data), # All columns
                           predictor_cols = NULL,         # Random predictors
                           missing_rate = rate)
    
    for (method in methods) {
      # 3. Impute missing values
      start_time <- Sys.time()
      
      if (method == "MIDAS") {
        # Special handling for MIDAS
        midas_input <- sprintf("temp_mcar_%.2f.csv", rate)
        midas_output_prefix <- sprintf("midas_%.2f", rate)
        
        # Save temporary file for Python processing
        write_csv(mar_data, midas_input)
        
        # Call Python script (assumes it's in same directory)
        system(sprintf("python MIDAS.Pipe.py %s %s", midas_input, midas_output_prefix))

        # Process and pool MIDAS imputations
        imputed_data <- process_midas_imputations(
          file_pattern = paste0(midas_output_prefix, "_imp_%d.csv"),
          output_file = paste0(midas_output_prefix, "_pooled.csv")
        )
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

    
      if (is.null(imputed_data)) next
      
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
        "Method: %-10s | Rate: %-4s | RMSE: %.4f | PFC: %.4f | Time: %.1fs\n",
        method, paste0(rate*100, "%"), rmse, pfc, time_taken
      ))
    }
  }
  
  # Save results
  write_csv(results, "imputation_results.csv")
  cat("\nResults saved to 'imputation_results.csv'\n")
  
  return(results)
}

# =============================================
# EXECUTE THE PIPELINE
# =============================================

# Example usage
results <- run_imputation_pipeline(
  data_path = "adult_sample_processed.csv",
  missing_rates = c(0.05, 0.10, 0.15),
  methods = c("MICE", "FAMD", "missForest", "MIDAS")
)

# Print final results
print(results)
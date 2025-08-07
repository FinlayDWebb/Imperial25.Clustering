# Fixed Imputation Evaluation Script
# Load required libraries
library(dplyr)
library(corrr)

# Function to calculate RMSE for numerical variables
calculate_rmse <- function(original, imputed) {
  # Select only numerical columns
  num_cols <- sapply(original, is.numeric)
  
  if (sum(num_cols) == 0) {
    return(NA)
  }
  
  original_num <- original[, num_cols, drop = FALSE]
  imputed_num <- imputed[, num_cols, drop = FALSE]
  
  # Calculate RMSE for each numerical column
  rmse_values <- sapply(names(original_num), function(col) {
    sqrt(mean((original_num[[col]] - imputed_num[[col]])^2, na.rm = TRUE))
  })
  
  # Return mean RMSE across all numerical columns
  return(mean(rmse_values, na.rm = TRUE))
}

# Function to calculate PFC for categorical variables
calculate_pfc <- function(original, imputed) {
  # Select only categorical columns (factors and characters)
  cat_cols <- sapply(original, function(x) is.factor(x) || is.character(x))
  
  if (sum(cat_cols) == 0) {
    return(NA)
  }
  
  original_cat <- original[, cat_cols, drop = FALSE]
  imputed_cat <- imputed[, cat_cols, drop = FALSE]
  
  # Calculate PFC for each categorical column
  pfc_values <- sapply(names(original_cat), function(col) {
    # Count mismatches
    mismatches <- sum(original_cat[[col]] != imputed_cat[[col]], na.rm = TRUE)
    # Total valid comparisons
    total <- sum(!is.na(original_cat[[col]]) & !is.na(imputed_cat[[col]]))
    
    if (total == 0) return(NA)
    return(mismatches / total)
  })
  
  # Return mean PFC across all categorical columns
  return(mean(pfc_values, na.rm = TRUE))
}

# Function to calculate correlation for numerical variables
calculate_correlation <- function(original, imputed) {
  # Select only numerical columns
  num_cols <- sapply(original, is.numeric)
  
  if (sum(num_cols) == 0) {
    return(NA)
  }
  
  original_num <- original[, num_cols, drop = FALSE]
  imputed_num <- imputed[, num_cols, drop = FALSE]
  
  # Calculate correlation for each numerical column
  cor_values <- sapply(names(original_num), function(col) {
    cor(original_num[[col]], imputed_num[[col]], use = "complete.obs")
  })
  
  # Return mean correlation across all numerical columns
  return(mean(cor_values, na.rm = TRUE))
}

# Load original dataset
print("Loading original dataset...")
original_data <- read.csv("adult_sample_processed.csv")
print(paste("Original data dimensions:", nrow(original_data), "rows,", ncol(original_data), "columns"))

# Define imputed datasets
imputed_files <- c(
  "adult_sample_mcar_FAMD.csv",
  "adult_sample_mnar_FAMD.csv",
  "adult_sample_mcar_MICE.csv",
  "adult_sample_mnar_MICE.csv",
  "adult_sample_mcar_MIDAS.csv",
  "adult_sample_mnar_MIDAS.csv",
  "adult_sample_mcar_missForest.csv",
  "adult_sample_mnar_missForest.csv",
  "adult_mcar_midas_imp_1.csv",
  "adult_mcar_midas_imp_2.csv",
  "adult_mcar_midas_imp_3.csv",
  "adult_mcar_midas_imp_4.csv",
  "adult_mcar_midas_imp_5.csv",
  "adult_mnar_midas_imp_1.csv",
  "adult_mnar_midas_imp_2.csv",
  "adult_mnar_midas_imp_3.csv",
  "adult_mnar_midas_imp_4.csv",
  "adult_mnar_midas_imp_5.csv"
)

# Check which files exist before processing
print("=== FILE EXISTENCE CHECK ===")
existing_files <- c()
for (file in imputed_files) {
  if (file.exists(file)) {
    print(paste("✓ Found:", file))
    existing_files <- c(existing_files, file)
  } else {
    print(paste("✗ Missing:", file))
  }
}

# Initialize results dataframe
results <- data.frame(
  Dataset = character(),
  Method = character(),
  Missing_Pattern = character(),
  RMSE = numeric(),
  PFC = numeric(),
  Correlation = numeric(),
  stringsAsFactors = FALSE
)

# Process each existing imputed dataset
print(paste("Processing", length(existing_files), "existing files..."))
success_count <- 0

for (file in existing_files) {
  print(paste("PROCESSING:", file))
  
  tryCatch({
    # Load imputed dataset
    imputed_data <- read.csv(file)
    print(paste("✓ Loaded. Dimensions:", nrow(imputed_data), "x", ncol(imputed_data)))
    
    # Verify dimensions match original data
    if (nrow(imputed_data) != nrow(original_data) || ncol(imputed_data) != ncol(original_data)) {
      stop(paste("Dimension mismatch! Original:", 
                 nrow(original_data), "x", ncol(original_data),
                 "Imputed:", nrow(imputed_data), "x", ncol(imputed_data)))
    }
    
    # ROBUST FILENAME PARSING
    file_base <- basename(file)
    if (grepl("sample", file_base)) {
      # adult_sample_<pattern>_<method>.csv
      parts <- unlist(strsplit(file_base, "_"))
      missing_pattern <- toupper(parts[3])
      method <- tools::file_path_sans_ext(parts[4])
    } else if (grepl("midas_imp", file_base)) {
      # adult_<pattern>_midas_imp_<num>.csv
      parts <- unlist(strsplit(file_base, "_"))
      missing_pattern <- toupper(parts[2])
      method <- "MIDAS"
    } else {
      stop("Unknown filename format")
    }
    
    # STANDARDIZE METHOD NAMES
    method <- case_when(
      tolower(method) %in% c("midas") ~ "MIDAS",
      tolower(method) %in% c("mice") ~ "MICE",
      tolower(method) %in% c("famd") ~ "FAMD",
      tolower(method) %in% c("missforest") ~ "missForest",
      TRUE ~ method
    )
    print(paste("Method:", method, "| Pattern:", missing_pattern))
    
    # SAFELY CALCULATE METRICS
    rmse <- tryCatch(calculate_rmse(original_data, imputed_data),
                     error = function(e) { print(paste("RMSE Error:", e$message)); NA })
    pfc <- tryCatch(calculate_pfc(original_data, imputed_data),
                    error = function(e) { print(paste("PFC Error:", e$message)); NA })
    correlation <- tryCatch(calculate_correlation(original_data, imputed_data),
                            error = function(e) { print(paste("Correlation Error:", e$message)); NA })
    
    # Add to results
    results <- rbind(results, data.frame(
      Dataset = file,
      Method = method,
      Missing_Pattern = missing_pattern,
      RMSE = ifelse(is.numeric(rmse), rmse, NA),
      PFC = ifelse(is.numeric(pfc), pfc, NA),
      Correlation = ifelse(is.numeric(correlation), correlation, NA)
    ))
    
    success_count <- success_count + 1
    print(paste("✓ Added. Success:", success_count, "/", length(existing_files)))
    
  }, error = function(e) {
    print(paste("⚠ PROCESSING FAILED:", e$message))
  })
  print("────────────────────────")
}

# SAVE RESULTS
if (nrow(results) > 0) {
  write.csv(results, "imputation_evaluation_results_FIXED.csv", row.names = FALSE)
  print(paste("✅ Saved results for", nrow(results), "files"))
} else {
  print("💥 No results to save!")
}

# Print final results
print("FINAL RESULTS:")
print(results)

# Print summary
if (nrow(results) > 0) {
  print("=== SUMMARY BY METHOD ===")
  summary_by_method <- results %>%
    group_by(Method) %>%
    summarise(
      Mean_RMSE = mean(RMSE, na.rm = TRUE),
      Mean_PFC = mean(PFC, na.rm = TRUE),
      Mean_Correlation = mean(Correlation, na.rm = TRUE),
      .groups = 'drop'
    )
  print(summary_by_method)
  
  print("=== SUMMARY BY MISSING PATTERN ===")
  summary_by_pattern <- results %>%
    group_by(Missing_Pattern) %>%
    summarise(
      Mean_RMSE = mean(RMSE, na.rm = TRUE),
      Mean_PFC = mean(PFC, na.rm = TRUE),
      Mean_Correlation = mean(Correlation, na.rm = TRUE),
      .groups = 'drop'
    )
  print(summary_by_pattern)
}

# Print interpretation guide
cat("\n=== INTERPRETATION GUIDE ===\n")
cat("RMSE (Root Mean Square Error):\n")
cat("  - Lower values = Better performance\n")
cat("  - Measures accuracy for numerical variables\n\n")
cat("PFC (Proportion of Falsely Classified):\n")
cat("  - Lower values = Better performance\n")
cat("  - Measures accuracy for categorical variables\n")
cat("  - Range: 0 to 1 (0 = perfect, 1 = completely wrong)\n\n")
cat("Correlation:\n")
cat("  - Higher values = Better performance\n")
cat("  - Measures linear relationship preservation\n")
cat("  - Range: -1 to 1 (1 = perfect positive correlation)\n")
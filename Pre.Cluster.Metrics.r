# Imputation Evaluation Script
# Calculate RMSE, PFC, and Correlation for imputed datasets

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
for (file in imputed_files) {
  if (file.exists(file)) {
    print(paste("✓ Found:", file))
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

# Process each imputed dataset
print("Processing imputed datasets...")
print(paste("Total files to process:", length(imputed_files)))

for (i in 1:length(imputed_files)) {
  file <- imputed_files[i]
  print(paste("Processing file", i, "of", length(imputed_files), ":", file))
  
  if (!file.exists(file)) {
    print(paste("WARNING: File not found:", file))
    next
  }
  
  tryCatch({
    imputed_data <- read.csv(file)
    print(paste("Successfully loaded:", file, "- Dimensions:", nrow(imputed_data), "x", ncol(imputed_data)))
    
      # FIXED FILENAME PARSING
    file_base <- gsub(".csv", "", file)
    file_parts <- strsplit(file_base, "_")[[1]]
    
    if ("sample" %in% file_parts) {
      # Standard naming: adult_sample_<pattern>_<method>
      missing_pattern <- toupper(file_parts[3])
      method <- file_parts[4]
    } else {
      # MIDAS individual naming: adult_<pattern>_midas_imp_<num>
      missing_pattern <- toupper(file_parts[2])
      method <- "MIDAS"  # Explicitly set to MIDAS
    }
    
   # Standardize method naming
    method <- case_when(
      tolower(method) == "midas" ~ "MIDAS",
      tolower(method) == "mice" ~ "MICE",
      tolower(method) == "famd" ~ "FAMD",
      tolower(method) == "missforest" ~ "missForest",
      TRUE ~ method
    )
    
    print(paste("Extracted - Method:", method, "Missing Pattern:", missing_pattern))

# Calculate metrics
    print("Calculating RMSE...")
    rmse <- calculate_rmse(original_data, imputed_data)
    print(paste("RMSE:", rmse))
    
    print("Calculating PFC...")
    pfc <- calculate_pfc(original_data, imputed_data)
    print(paste("PFC:", pfc))
    
    print("Calculating Correlation...")
    correlation <- calculate_correlation(original_data, imputed_data)
    print(paste("Correlation:", correlation))
    
# Add to results
    results <- rbind(results, data.frame(
      Dataset = file,
      Method = method,
      Missing_Pattern = missing_pattern,
      RMSE = rmse,
      PFC = pfc,
      Correlation = correlation
    ))
    
    print(paste("Added results for:", file))
    
  }, error = function(e) {
    print(paste("ERROR processing file:", file))
    print(paste("Error message:", e$message))
  })
  print("---")
}

# Display results
print("=== IMPUTATION EVALUATION RESULTS ===")
print(paste("Total datasets processed:", nrow(results)))
print(results)

# Show which files were processed vs expected
print("\n=== PROCESSING SUMMARY ===")
processed_files <- results$Dataset
expected_files <- imputed_files
missing_files <- setdiff(expected_files, processed_files)

print(paste("Expected files:", length(expected_files)))
print(paste("Successfully processed:", length(processed_files)))

if (length(missing_files) > 0) {
  print("Files that were NOT processed:")
  for (file in missing_files) {
    print(paste("  -", file))
  }
}

# Create summary by method
print("\n=== SUMMARY BY METHOD ===")
summary_by_method <- results %>%
  group_by(Method) %>%
  summarise(
    Mean_RMSE = mean(RMSE, na.rm = TRUE),
    Mean_PFC = mean(PFC, na.rm = TRUE),
    Mean_Correlation = mean(Correlation, na.rm = TRUE),
    .groups = 'drop'
  )
print(summary_by_method)

# Create summary by missing pattern
print("\n=== SUMMARY BY MISSING PATTERN ===")
summary_by_pattern <- results %>%
  group_by(Missing_Pattern) %>%
  summarise(
    Mean_RMSE = mean(RMSE, na.rm = TRUE),
    Mean_PFC = mean(PFC, na.rm = TRUE),
    Mean_Correlation = mean(Correlation, na.rm = TRUE),
    .groups = 'drop'
  )
print(summary_by_pattern)

# Save results to CSV
write.csv(results, "imputation_evaluation_results.csv", row.names = FALSE)
print("\nResults saved to: imputation_evaluation_results.csv")

# FIXED Interpretation Guide
print("\n=== INTERPRETATION GUIDE ===")
print("RMSE (Root Mean Square Error):")
print("  - Lower values = Better performance")
print("  - Measures accuracy for numerical variables")
print("")
print("PFC (Proportion of Falsely Classified):")
print("  - Lower values = Better performance")
print("  - Measures accuracy for categorical variables")
print("  - Range: 0 to 1 (0 = perfect, 1 = completely wrong)")
print("")
print("Correlation:")
print("  - Higher values = Better performance")
print("  - Measures linear relationship preservation")
print("  - Range: -1 to 1 (1 = perfect positive correlation)")
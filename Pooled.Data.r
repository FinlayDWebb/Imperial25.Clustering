
# First I'm going to pool the MIDAS imputed datasets together, by aggregating the imputed values.

# Pool MIDAS Multiple Imputation Datasets
# This script pools multiple imputed datasets from MIDAS into single datasets

# Function to pool multiple imputed datasets
pool_imputed_datasets <- function(file_pattern, output_filename) {
  
  # Find all files matching the pattern
  files <- list.files(pattern = file_pattern, full.names = TRUE)
  
  if(length(files) == 0) {
    stop(paste("No files found matching pattern:", file_pattern))
  }
  
  cat("Found", length(files), "files to pool:\n")
  print(files)
  
  # Read all datasets
  datasets <- list()
  for(i in 1:length(files)) {
    cat("Reading", files[i], "\n")
    datasets[[i]] <- read.csv(files[i], stringsAsFactors = FALSE)
  }
  
  # Check dimensions are consistent
  dims <- sapply(datasets, dim)
  if(!all(dims[1,] == dims[1,1]) || !all(dims[2,] == dims[2,1])) {
    stop("Datasets have inconsistent dimensions!")
  }
  
  cat("All datasets have dimensions:", dims[1,1], "x", dims[2,1], "\n")
  
  # Initialize pooled dataset with first dataset structure
  pooled_data <- datasets[[1]]
  
  # Identify numeric columns (these will be averaged)
  numeric_cols <- sapply(pooled_data, is.numeric)
  cat("Numeric columns to be averaged:", names(pooled_data)[numeric_cols], "\n")
  
  # Identify categorical columns (these will use mode/most frequent)
  categorical_cols <- !numeric_cols
  cat("Categorical columns (mode used):", names(pooled_data)[categorical_cols], "\n")
  
  # Pool numeric columns by averaging
  for(col in names(pooled_data)[numeric_cols]) {
    # Create matrix of values from all datasets
    values_matrix <- sapply(datasets, function(x) x[[col]])
    
    # Average across imputations (by row)
    pooled_data[[col]] <- rowMeans(values_matrix, na.rm = TRUE)
  }
  
  # Pool categorical columns by mode (most frequent value)
  for(col in names(pooled_data)[categorical_cols]) {
    pooled_values <- c()
    
    for(row in 1:nrow(pooled_data)) {
      # Get values from all datasets for this row
      row_values <- sapply(datasets, function(x) x[row, col])
      
      # Find mode (most frequent value)
      mode_value <- names(sort(table(row_values), decreasing = TRUE))[1]
      pooled_values[row] <- mode_value
    }
    
    pooled_data[[col]] <- pooled_values
  }
  
  # Save pooled dataset
  write.csv(pooled_data, output_filename, row.names = FALSE)
  cat("Pooled dataset saved as:", output_filename, "\n")
  
  # Return summary info
  return(list(
    n_datasets = length(datasets),
    dimensions = dim(pooled_data),
    numeric_cols = sum(numeric_cols),
    categorical_cols = sum(categorical_cols),
    output_file = output_filename
  ))
}

# ===============================
# Pool MCAR MIDAS datasets
# ===============================

cat("=== POOLING MCAR MIDAS DATASETS ===\n")
mcar_summary <- pool_imputed_datasets(
  file_pattern = "adult_mcar_midas_imp_[0-9]+\\.csv",
  output_filename = "adult_mcar_midas_pooled.csv"
)

print(mcar_summary)

# ===============================
# Pool MNAR MIDAS datasets  
# ===============================

cat("\n=== POOLING MNAR MIDAS DATASETS ===\n")
mnar_summary <- pool_imputed_datasets(
  file_pattern = "adult_mnar_midas_imp_[0-9]+\\.csv", 
  output_filename = "adult_mnar_midas_pooled.csv"
)

print(mnar_summary)

# ===============================
# Verification and Summary
# ===============================

cat("\n=== VERIFICATION ===\n")

# Check if pooled files were created successfully
if(file.exists("adult_mcar_midas_pooled.csv")) {
  mcar_pooled <- read.csv("adult_mcar_midas_pooled.csv", stringsAsFactors = FALSE)
  cat("MCAR pooled dataset dimensions:", dim(mcar_pooled), "\n")
  cat("MCAR pooled dataset summary:\n")
  print(summary(mcar_pooled))
} else {
  cat("ERROR: MCAR pooled dataset not created!\n")
}

if(file.exists("adult_mnar_midas_pooled.csv")) {
  mnar_pooled <- read.csv("adult_mnar_midas_pooled.csv", stringsAsFactors = FALSE)
  cat("\nMNAR pooled dataset dimensions:", dim(mnar_pooled), "\n")
  cat("MNAR pooled dataset summary:\n")
  print(summary(mnar_pooled))
} else {
  cat("ERROR: MNAR pooled dataset not created!\n")
}

# Create final summary
cat("\n=== FINAL SUMMARY ===\n")
cat("Original files processed:\n")
cat("- MCAR MIDAS files:", mcar_summary$n_datasets, "datasets\n")
cat("- MNAR MIDAS files:", mnar_summary$n_datasets, "datasets\n")
cat("\nOutput files created:\n")
cat("- adult_mcar_midas_pooled.csv\n")
cat("- adult_mnar_midas_pooled.csv\n")
cat("\nPooling method:\n")
cat("- Numeric columns: Averaged across imputations\n")
cat("- Categorical columns: Mode (most frequent value) across imputations\n")
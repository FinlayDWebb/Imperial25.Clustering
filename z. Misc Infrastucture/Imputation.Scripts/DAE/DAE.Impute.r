# Not working yet.


# CLEAN START - Remove existing installations first
# Create fresh virtual environment
reticulate::virtualenv_remove("r-reticulate")  # Remove if exists
reticulate::virtualenv_create("r-reticulate")

# Install ONLY compatible versions (don't mix versions!)
reticulate::py_install("tensorflow==2.12.0")  # Stick to one TF version
reticulate::py_install("numpy==1.23.5")       # Compatible numpy
reticulate::py_install("pandas")
reticulate::py_install("matplotlib")

# DO NOT install keras separately - it comes with TensorFlow
# DO NOT install tensorflow_addons unless specifically needed

# Restart R session to clear any cached modules
.rs.restartR()  # Uncomment this line and run it, then re-run the rest

# Install and load required packages
if (!require("rMIDAS", quietly = TRUE)) {
  if (!require("remotes")) install.packages("remotes")
  remotes::install_github("MIDASverse/rMIDAS")
}
library(rMIDAS)
library(dplyr)

set.seed(42)  # For reproducibility

# Function to process and impute data
impute_with_MIDAS <- function(input_file, output_file) {
  # Read data
  data <- read.csv(input_file, stringsAsFactors = FALSE)
  
  # Debug: Check data structure
  cat("Data dimensions:", dim(data), "\n")
  cat("Column names:", names(data), "\n")
  cat("Data types:\n")
  print(sapply(data, class))
  
  # Check for missing values
  cat("Missing values per column:\n")
  print(sapply(data, function(x) sum(is.na(x))))
  
  # Identify variable types based on your data
  # You may need to adjust these based on your actual data structure
  
  # For binary variables, ensure they are 0/1 or have exactly 2 unique values
  # Convert factors to numeric if needed
  if("sex" %in% names(data)) {
    if(is.factor(data$sex) || is.character(data$sex)) {
      data$sex <- as.numeric(as.factor(data$sex)) - 1  # Convert to 0/1
    }
  }
  
  if("income" %in% names(data)) {
    if(is.factor(data$income) || is.character(data$income)) {
      data$income <- as.numeric(as.factor(data$income)) - 1  # Convert to 0/1
    }
  }
  
  # Identify continuous variables (adjust based on your data)
  numeric_cols <- sapply(data, is.numeric)
  unique_counts <- sapply(data, function(x) length(unique(x[!is.na(x)])))
  
  # Automatically identify variable types
  cont_vars <- names(data)[numeric_cols & unique_counts > 10]
  bin_vars <- names(data)[unique_counts == 2]
  cat_vars <- names(data)[unique_counts > 2 & unique_counts <= 10 & !names(data) %in% cont_vars]
  
  cat("Continuous variables:", cont_vars, "\n")
  cat("Binary variables:", bin_vars, "\n")
  cat("Categorical variables:", cat_vars, "\n")
  
  # Prepare data for rMIDAS - CORRECTED APPROACH
  converted_data <- convert(
    data,
    bin_cols = bin_vars,              # Binary variables
    cat_cols = cat_vars,              # Categorical variables
    minmax_scale = TRUE               # Scale continuous variables
  )
  
  # Train MIDAS model
  cat("Training MIDAS model...\n")
  model <- train(
    converted_data,
    training_epochs = 50,
    layer_structure = c(128, 64),
    input_drop = 0.7,
    seed = 42
  )
  
  # Generate imputation using the complete() function - THIS IS ESSENTIAL!
  cat("Generating imputations...\n")
  imputed_data <- complete(model, m = 1)  # Generate 1 imputed dataset
  
  # Extract the imputed dataset (complete() returns a list)
  final_data <- imputed_data[[1]]
  
  # Verify imputation worked
  cat("Original missing values:\n")
  print(sapply(data, function(x) sum(is.na(x))))
  cat("Imputed missing values:\n")
  print(sapply(final_data, function(x) sum(is.na(x))))
  
  # Save results
  write.csv(final_data, output_file, row.names = FALSE)
  cat("Imputation completed for", input_file, "\nSaved as", output_file, "\n")
  
  return(final_data)
}

# Test Python setup before proceeding
cat("Testing Python setup...\n")
reticulate::py_run_string("import tensorflow as tf; print('TensorFlow version:', tf.__version__)")
reticulate::py_run_string("print('TensorFlow built with CUDA:', tf.test.is_built_with_cuda())")

# Process both datasets
tryCatch({
  result1 <- impute_with_MIDAS("adult_sample_mcar.csv", "adult_sample_mcar_MIDAS.csv")
  result2 <- impute_with_MIDAS("adult_sample_mnar.csv", "adult_sample_mnar_MIDAS.csv")
}, error = function(e) {
  cat("Error occurred:", e$message, "\n")
  cat("Please check your data format and variable types\n")
})

# Alternative approach if you want multiple imputations (recommended)
impute_with_MIDAS_multiple <- function(input_file, output_prefix, m = 5) {
  # Read data
  data <- read.csv(input_file, stringsAsFactors = FALSE)
  
  # Process data types as above...
  # [Include the same data processing steps as in the main function]
  
  # Convert data
  converted_data <- convert(
    data,
    bin_cols = bin_vars,
    cat_cols = cat_vars,
    minmax_scale = TRUE
  )
  
  # Train model
  model <- train(
    converted_data,
    training_epochs = 50,
    layer_structure = c(128, 64),
    input_drop = 0.7,
    seed = 42
  )
  
  # Generate multiple imputations
  imputed_data <- complete(model, m = m)
  
  # Save each imputed dataset
  for(i in 1:m) {
    output_file <- paste0(output_prefix, "_imp_", i, ".csv")
    write.csv(imputed_data[[i]], output_file, row.names = FALSE)
  }
  
  return(imputed_data)
}

# Example usage for multiple imputations:
# multiple_imps <- impute_with_MIDAS_multiple("adult_sample_mcar.csv", "adult_mcar", m = 5)
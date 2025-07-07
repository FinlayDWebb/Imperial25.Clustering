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
  
  # Prepare data for rMIDAS - CORRECTED APPROACH
  converted_data <- convert(
    data,
    bin_cols = c("sex", "income"),              # Binary variables
    cat_cols = c("marital_status", "workclass", "occupation"),  # Categorical variables
    minmax_scale = TRUE                         # Scale continuous variables
  )
  
  # Train MIDAS model
  model <- train(
    converted_data,
    training_epochs = 50,
    layer_structure = c(128, 64),
    input_drop = 0.7,
    seed = 42
  )
  
  # Generate single imputation (m=1)
  imputed_data <- complete(model, m = 1)
  
  # Extract the imputed dataset
  final_data <- imputed_data[[1]]
  
  # Save results
  write.csv(final_data, output_file, row.names = FALSE)
  cat("Imputation completed for", input_file, "\nSaved as", output_file, "\n")
}

# Test Python setup before proceeding
cat("Testing Python setup...\n")
reticulate::py_run_string("import tensorflow as tf; print('TensorFlow version:', tf.__version__)")
reticulate::py_run_string("print('TensorFlow built with CUDA:', tf.test.is_built_with_cuda())")

# Process both datasets
impute_with_MIDAS("adult_sample_mcar.csv", "adult_sample_mcar_MIDAS.csv")
impute_with_MIDAS("adult_sample_mnar.csv", "adult_sample_mnar_MIDAS.csv")
# Install Python dependencies if needed
reticulate::py_install("tensorflow")
reticulate::py_install("matplotlib")
reticulate::py_install("numpy")
reticulate::py_install("pandas")

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

# Process both datasets
impute_with_MIDAS("adult_sample_mcar.csv", "adult_sample_mcar_MIDAS.csv")
impute_with_MIDAS("adult_sample_mnar.csv", "adult_sample_mnar_MIDAS.csv")
library(mice) # Multiple Imputation by Chained Equations
library(missMDA) # Principal Component Analysis for Missing Data
library(missForest) # Random Forest for Missing Data
library(rMIDAS) # DAE for Missing Data

### We start with MICE

# Load required libraries
library(mice)
library(readr)
library(dplyr)

# Set seed for reproducibility
set.seed(42)

# Function to set up MICE methods based on variable types
setup_mice_methods <- function(data) {
  methods <- character(ncol(data))
  names(methods) <- names(data)
  
  for (col in names(data)) {
    if (is.numeric(data[[col]])) {
      methods[col] <- "pmm"  # Predictive mean matching for continuous
    } else if (is.factor(data[[col]]) || is.character(data[[col]])) {
      unique_vals <- length(unique(data[[col]][!is.na(data[[col]])]))
      if (unique_vals == 2) {
        methods[col] <- "logreg"  # Logistic regression for binary
      } else {
        methods[col] <- "polyreg"  # Polytomous logistic regression for multilevel
      }
    }
  }
  
  return(methods)
}

# Function to perform MICE imputation
perform_mice_imputation <- function(data, dataset_name, m = 5, maxit = 10) {
  cat("\n=== MICE Imputation for", dataset_name, "===\n")
  
  # Convert character variables to factors for better imputation
  data_for_mice <- data %>%
    mutate_if(is.character, as.factor)
  
  # Set up methods
  methods <- setup_mice_methods(data_for_mice)
  cat("Imputation methods:\n")
  print(methods[methods != ""])
  
  # Perform MICE imputation
  cat("\nPerforming MICE imputation...\n")
  mice_result <- mice(
    data = data_for_mice,
    m = m,                    # Number of imputations
    method = methods,         # Methods for each variable
    maxit = maxit,           # Maximum iterations
    printFlag = TRUE,        # Print progress
    seed = 42               # For reproducibility
  )
  
  # Complete the data (using first imputation)
  completed_data <- complete(mice_result, 1)
  
  # Convert factors back to characters for consistency
  completed_data <- completed_data %>%
    mutate_if(is.factor, as.character)
  
  return(completed_data)
}

# Main execution
cat("=== MICE Imputation Analysis ===\n")

# Load datasets
cat("Loading datasets...\n")
mcar_data <- read_csv("adult_sample_mcar.csv")
mnar_data <- read_csv("adult_sample_mnar.csv")

# Perform MICE imputation on MCAR data
mcar_imputed <- perform_mice_imputation(mcar_data, "MCAR", m = 5, maxit = 10)

# Perform MICE imputation on MNAR data
mnar_imputed <- perform_mice_imputation(mnar_data, "MNAR", m = 5, maxit = 10)

# Save imputed datasets with MICE label
cat("\nSaving imputed datasets...\n")
write_csv(mcar_imputed, "adult_sample_mcar_MICE.csv")
write_csv(mnar_imputed, "adult_sample_mnar_MICE.csv")

# Summary
cat("\n=== Script Complete ===\n")
cat("Output files created:\n")
cat("- adult_sample_mcar_MICE.csv: MCAR dataset imputed with MICE\n")
cat("- adult_sample_mnar_MICE.csv: MNAR dataset imputed with MICE\n")

# Quick verification
cat("\nVerification - Missing values after imputation:\n")
cat("MCAR dataset:", sum(is.na(mcar_imputed)), "missing values\n")
cat("MNAR dataset:", sum(is.na(mnar_imputed)), "missing values\n")

####################################################################################################################

### Now we move to missMDA

# Function to perform MIFAMD imputation
perform_mifamd_imputation <- function(data, dataset_name, ncp = 2, nboot = 20, method = "Regularized") {
  cat("\n=== MIFAMD Imputation for", dataset_name, "===\n")
  
  # Convert character variables to factors for MIFAMD
  data_for_mifamd <- data %>%
    mutate_if(is.character, as.factor)
  
  # Show data types
  cat("Data types for imputation:\n")
  print(sapply(data_for_mifamd, class))
  
  # Perform MIFAMD imputation
  cat("\nPerforming MIFAMD imputation...\n")
  cat("Parameters: ncp =", ncp, ", method =", method, ", nboot =", nboot, "\n")
  
  mifamd_result <- MIFAMD(
    X = data_for_mifamd,
    ncp = ncp,                # Number of components
    method = method,          # "Regularized" or "EM"
    coeff.ridge = 1,         # Regularization coefficient
    threshold = 1e-06,       # Convergence threshold
    seed = 42,               # For reproducibility
    maxiter = 1000,          # Maximum iterations
    nboot = nboot,           # Number of imputed datasets
    verbose = TRUE           # Print progress
  )
  
  # Get the first imputed dataset
  completed_data <- mifamd_result$res.MI[[1]]
  
  # Convert factors back to characters for consistency
  completed_data <- completed_data %>%
    mutate_if(is.factor, as.character)
  
  return(completed_data)
}

# Main execution
cat("=== MIFAMD Imputation Analysis ===\n")

# Load datasets
cat("Loading datasets...\n")
mcar_data <- read_csv("adult_sample_mcar.csv")
mnar_data <- read_csv("adult_sample_mnar.csv")

# Perform MIFAMD imputation on MCAR data
mcar_imputed <- perform_mifamd_imputation(mcar_data, "MCAR", ncp = 2, nboot = 20, method = "Regularized")

# Perform MIFAMD imputation on MNAR data
mnar_imputed <- perform_mifamd_imputation(mnar_data, "MNAR", ncp = 2, nboot = 20, method = "Regularized")

# Save imputed datasets with MIFAMD label
cat("\nSaving imputed datasets...\n")
write_csv(mcar_imputed, "adult_sample_mcar_MIFAMD.csv")
write_csv(mnar_imputed, "adult_sample_mnar_MIFAMD.csv")

# Summary
cat("\n=== Script Complete ===\n")
cat("Output files created:\n")
cat("- adult_sample_mcar_MIFAMD.csv: MCAR dataset imputed with MIFAMD\n")
cat("- adult_sample_mnar_MIFAMD.csv: MNAR dataset imputed with MIFAMD\n")

# Quick verification
cat("\nVerification - Missing values after imputation:\n")
cat("MCAR dataset:", sum(is.na(mcar_imputed)), "missing values\n")
cat("MNAR dataset:", sum(is.na(mnar_imputed)), "missing values\n")

####################################################################################################################

### Now we move to missForest

# Function to perform missForest imputation
perform_missforest_imputation <- function(data, dataset_name, maxiter = 10, ntree = 100) {
  cat("\n=== missForest Imputation for", dataset_name, "===\n")
  
  # Convert character variables to factors for missForest
  data_for_missforest <- data %>%
    mutate_if(is.character, as.factor)
  
  # Show data types
  cat("Data types for imputation:\n")
  print(sapply(data_for_missforest, class))
  
  # Perform missForest imputation
  cat("\nPerforming missForest imputation...\n")
  cat("Parameters: maxiter =", maxiter, ", ntree =", ntree, "\n")
  
  missforest_result <- missForest(
    xmis = data_for_missforest,
    maxiter = maxiter,       # Maximum iterations
    ntree = ntree,           # Number of trees per forest
    variablewise = FALSE,    # Don't return OOB error for each variable
    decreasing = FALSE,      # Sort variables by increasing missing values
    verbose = TRUE,          # Print progress
    mtry = floor(sqrt(ncol(data_for_missforest))),  # Default mtry
    replace = TRUE,          # Bootstrap sampling with replacement
    parallelize = 'no'       # No parallelization
  )
  
  # Get the imputed dataset
  completed_data <- missforest_result$ximp
  
  # Convert factors back to characters for consistency
  completed_data <- completed_data %>%
    mutate_if(is.factor, as.character)
  
  return(completed_data)
}

# Main execution
cat("=== missForest Imputation Analysis ===\n")

# Load datasets
cat("Loading datasets...\n")
mcar_data <- read_csv("adult_sample_mcar.csv")
mnar_data <- read_csv("adult_sample_mnar.csv")

# Perform missForest imputation on MCAR data
mcar_imputed <- perform_missforest_imputation(mcar_data, "MCAR", maxiter = 10, ntree = 100)

# Perform missForest imputation on MNAR data
mnar_imputed <- perform_missforest_imputation(mnar_data, "MNAR", maxiter = 10, ntree = 100)

# Save imputed datasets with missForest label
cat("\nSaving imputed datasets...\n")
write_csv(mcar_imputed, "adult_sample_mcar_missForest.csv")
write_csv(mnar_imputed, "adult_sample_mnar_missForest.csv")

# Summary
cat("\n=== Script Complete ===\n")
cat("Output files created:\n")
cat("- adult_sample_mcar_missForest.csv: MCAR dataset imputed with missForest\n")
cat("- adult_sample_mnar_missForest.csv: MNAR dataset imputed with missForest\n")

# Quick verification
cat("\nVerification - Missing values after imputation:\n")
cat("MCAR dataset:", sum(is.na(mcar_imputed)), "missing values\n")
cat("MNAR dataset:", sum(is.na(mnar_imputed)), "missing values\n")

####################################################################################################################

### Now we move to rMIDAS

# Function to perform rMIDAS imputation
perform_rmidas_imputation <- function(data, dataset_name, training_epochs = 20, layer_structure = c(128, 128)) {
  cat("\n=== rMIDAS Imputation for", dataset_name, "===\n")
  
  # Show data info
  cat("Data dimensions:", dim(data), "\n")
  cat("Missing values:", sum(is.na(data)), "\n")
  
  # Convert data for MIDAS (preprocessing)
  cat("Converting data for MIDAS...\n")
  data_converted <- convert(data, bin_cols = c("sex", "income"))
  
  # Train the MIDAS model
  cat("Training MIDAS model...\n")
  cat("Parameters: epochs =", training_epochs, ", layer_structure =", paste(layer_structure, collapse = ","), "\n")
  
  midas_model <- train(
    data_converted,
    training_epochs = training_epochs,
    layer_structure = layer_structure,
    input_drop = 0.75,      # Input dropout rate
    seed = 42               # For reproducibility
  )
  
  # Generate completed dataset
  cat("Generating completed dataset...\n")
  completed_list <- complete(midas_model, m = 1)  # Generate 1 completed dataset
  completed_data <- completed_list[[1]]
  
  return(completed_data)
}

# Main execution
cat("=== rMIDAS Imputation Analysis ===\n")

# Check if Python environment is set up
cat("Note: rMIDAS requires Python. If this is your first time using rMIDAS,\n")
cat("you may need to set up the Python environment when prompted.\n\n")

# Load datasets
cat("Loading datasets...\n")
mcar_data <- read_csv("adult_sample_mcar.csv")
mnar_data <- read_csv("adult_sample_mnar.csv")

# Perform rMIDAS imputation on MCAR data
tryCatch({
  mcar_imputed <- perform_rmidas_imputation(mcar_data, "MCAR", training_epochs = 20, layer_structure = c(128, 128))
  
  # Save imputed dataset
  write_csv(mcar_imputed, "adult_sample_mcar_rMIDAS.csv")
  cat("MCAR imputation completed and saved.\n")
  
}, error = function(e) {
  cat("Error in MCAR imputation:", e$message, "\n")
  cat("This might be due to Python environment setup issues.\n")
})

# Perform rMIDAS imputation on MNAR data
tryCatch({
  mnar_imputed <- perform_rmidas_imputation(mnar_data, "MNAR", training_epochs = 20, layer_structure = c(128, 128))
  
  # Save imputed dataset
  write_csv(mnar_imputed, "adult_sample_mnar_rMIDAS.csv")
  cat("MNAR imputation completed and saved.\n")
  
}, error = function(e) {
  cat("Error in MNAR imputation:", e$message, "\n")
  cat("This might be due to Python environment setup issues.\n")
})

# Summary
cat("\n=== Script Complete ===\n")
cat("Expected output files:\n")
cat("- adult_sample_mcar_rMIDAS.csv: MCAR dataset imputed with rMIDAS\n")
cat("- adult_sample_mnar_rMIDAS.csv: MNAR dataset imputed with rMIDAS\n")

# Quick verification (if imputation succeeded)
if (exists("mcar_imputed")) {
  cat("\nVerification - Missing values after MCAR imputation:", sum(is.na(mcar_imputed)), "\n")
}
if (exists("mnar_imputed")) {
  cat("Verification - Missing values after MNAR imputation:", sum(is.na(mnar_imputed)), "\n")
}

# Additional notes
cat("\n=== Important Notes ===\n")
cat("1. rMIDAS requires Python (3.6-3.10) to be installed\n")
cat("2. On first use, you may be prompted to set up the Python environment\n")
cat("3. Training epochs and layer structure can be adjusted based on your data\n")
cat("4. The script uses binary column specification for 'sex' and 'income'\n")
cat("5. Input dropout rate is set to 0.75 (default recommendation)\n")

####################################################################################################################

# This should print all the imputed datasets for each method.
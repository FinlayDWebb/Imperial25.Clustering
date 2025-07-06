# Missing Data Imputation Comparison
# Methods: MICE, MIFAMD (missMDA), missForest, rMIDAS
# Datasets: adult_sample_mcar.csv, adult_sample_mnar.csv

# Load required libraries
library(mice)        # Multiple Imputation by Chained Equations
library(missMDA)     # Principal Component Analysis for Missing Data
library(missForest)  # Random Forest for Missing Data
library(rMIDAS)      # DAE for Missing Data
library(readr)       # Reading CSV files
library(dplyr)       # Data manipulation
library(purrr)       # Functional programming
library(readr)       # For reading CSV files

# Set global seed for reproducibility
set.seed(42)

# ----------------------------
# 1. Data Loading
# ----------------------------
cat("=== Loading Datasets ===\n")
mcar_data <- read_csv("adult_sample_mcar.csv", show_col_types = FALSE)
mnar_data <- read_csv("adult_sample_mnar.csv", show_col_types = FALSE)

cat("Missing values in original datasets:\n")
cat("- MCAR:", sum(is.na(mcar_data)), "missing values\n")
cat("- MNAR:", sum(is.na(mnar_data)), "missing values\n\n")

# ----------------------------
# 2. Imputation Functions
# ----------------------------

# ----- MICE Imputation -----
perform_mice_imputation <- function(data, dataset_name, m = 5, maxit = 10) {
  cat("\n=== MICE Imputation for", dataset_name, "===\n")
  
  # Convert characters to factors
  data <- data %>% mutate_if(is.character, as.factor)
  
  # Setup methods based on variable types
  setup_methods <- function(data) {
    methods <- character(ncol(data))
    names(methods) <- names(data)
    
    for (col in names(data)) {
      if (is.numeric(data[[col]])) {
        methods[col] <- "pmm"  # Predictive mean matching for continuous
      } else if (is.factor(data[[col]])) {
        unique_vals <- nlevels(data[[col]])
        if (unique_vals == 2) {
          methods[col] <- "logreg"  # Logistic regression for binary
        } else {
          methods[col] <- "polyreg"  # Polytomous regression for categorical
        }
      }
    }
    return(methods)
  }
  
  methods <- setup_methods(data)
  cat("Imputation methods:\n")
  print(methods[methods != ""])
  
  # Perform MICE
  mice_result <- mice::mice(
    data = data,
    m = m,
    method = methods,
    maxit = maxit,
    printFlag = TRUE,
    seed = 42
  )
  
  # Return first imputation
  mice::complete(mice_result, 1)
}

# ----- MIFAMD Imputation -----
perform_mifamd_imputation <- function(data, dataset_name, ncp = 2, nboot = 20) {
  cat("\n=== MIFAMD Imputation for", dataset_name, "===\n")
  
  # Convert characters to factors
  data <- data %>% mutate_if(is.character, as.factor)
  
  # Show data types
  cat("Data types:\n")
  print(sapply(data, class))
  
  # Perform MIFAMD
  cat("Performing MIFAMD (ncp =", ncp, ")...\n")
  mifamd_result <- missMDA::MIFAMD(
    X = data,
    ncp = ncp,
    method = "Regularized",
    seed = 42,
    nboot = nboot
  )
  
  # Return first imputation
  mifamd_result$res.MI[[1]]
}

# ----- missForest Imputation -----
perform_missforest_imputation <- function(data, dataset_name, maxiter = 10, ntree = 100) {
  cat("\n=== missForest Imputation for", dataset_name, "===\n")
  
  # Convert characters to factors
  data <- data %>% mutate_if(is.character, as.factor)
  
  # Show data types
  cat("Data types:\n")
  print(sapply(data, class))
  
  # Perform missForest
  cat("Performing missForest...\n")
  forest_result <- missForest::missForest(
    xmis = data,
    maxiter = maxiter,
    ntree = ntree,
    verbose = TRUE
  )
  
  # Return imputed data
  forest_result$ximp
}

# ----- rMIDAS Imputation -----
perform_rmidas_imputation <- function(data, dataset_name, training_epochs = 20, layer_structure = c(128, 128)) {
  cat("\n=== rMIDAS Imputation for", dataset_name, "===\n")
  
  # Convert characters to factors
  data <- data %>% mutate_if(is.character, as.factor)
  
  # Identify variable types
  factor_cols <- names(data)[sapply(data, is.factor)]
  binary_vars <- factor_cols[sapply(data[factor_cols], \(x) nlevels(x) == 2)]
  categorical_vars <- factor_cols[sapply(data[factor_cols], \(x) nlevels(x) > 2)]
  
  cat("Binary variables:", paste(binary_vars, collapse = ", "), "\n")
  cat("Categorical variables:", paste(categorical_vars, collapse = ", "), "\n")
  
  # Convert data for MIDAS
  cat("Converting data for rMIDAS...\n")
  data_converted <- rMIDAS::convert(
    data,
    bin_cols = binary_vars,
    cat_cols = categorical_vars
  )
  
  # Train model
  cat("Training MIDAS model...\n")
  midas_model <- rMIDAS::train(
    data_converted,
    training_epochs = training_epochs,
    layer_structure = layer_structure,
    input_drop = 0.75,
    seed = 42
  )
  
  # Generate imputations
  cat("Generating completed data...\n")
  completed_list <- rMIDAS::complete(midas_model, m = 1)
  completed_list[[1]]
}

# ----------------------------
# 3. Imputation Execution
# ----------------------------

# Define methods to run
methods <- list(
  "MICE" = perform_mice_imputation,
  "MIFAMD" = perform_mifamd_imputation,
  "missForest" = perform_missforest_imputation,
  "rMIDAS" = perform_rmidas_imputation
)

# Run imputations for both datasets
results <- list()

for (dataset in c("mcar", "mnar")) {
  dataset_name <- toupper(dataset)
  data <- get(paste0(dataset, "_data"))
  
  for (method_name in names(methods)) {
    cat("\n\n", rep("=", 50), "\n")
    cat("STARTING:", method_name, "on", dataset_name, "data\n")
    cat(rep("=", 50), "\n")
    
    # Run imputation with error handling
    result <- tryCatch({
      imputed <- methods[[method_name]](data, dataset_name)
      
      # Check for remaining missing values
      remaining_na <- sum(is.na(imputed))
      if (remaining_na > 0) {
        warning(method_name, " on ", dataset_name, 
                " still has ", remaining_na, " missing values")
      }
      
      imputed
    }, error = function(e) {
      cat("!! ERROR in ", method_name, " on ", dataset_name, ": ", e$message, "\n")
      NULL
    })
    
    # Store and save results
    if (!is.null(result)) {
      # Save to file
      filename <- paste0("adult_sample_", dataset, "_", method_name, ".csv")
      write_csv(result, filename)
      cat("Saved:", filename, "\n")
      
      # Store for verification
      results[[paste0(dataset, "_", method_name)]] <- result
    }
  }
}

# ----------------------------
# 4. Results Verification
# ----------------------------
cat("\n\n=== Imputation Results Summary ===")

if (length(results) > 0) {
  cat("\nMissing values after imputation:\n")
  for (result_name in names(results)) {
    na_count <- sum(is.na(results[[result_name]]))
    cat(sprintf("- %-25s: %d missing values\n", result_name, na_count))
  }
} else {
  cat("\nNo successful imputations were completed")
}

cat("\n=== Script Execution Complete ===\n")
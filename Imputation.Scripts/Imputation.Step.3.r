# Missing Data Imputation Comparison - Revised
# Methods: MICE, FAMD (missMDA), missForest, rMIDAS
# Datasets: adult_sample_mcar.csv, adult_sample_mnar.csv

# Load required libraries
library(mice)        # Multiple Imputation by Chained Equations
library(missMDA)     # Principal Component Analysis for Missing Data
library(missForest)  # Random Forest for Missing Data
library(rMIDAS)      # DAE for Missing Data
library(readr)       # Reading CSV files
library(dplyr)       # Data manipulation

# Set global seed for reproducibility
set.seed(42)

# ----------------------------
# 1. Data Loading with Type Specification
# ----------------------------
cat("=== Loading Datasets with Type Specification ===\n")
mcar_data <- read_csv("adult_sample_mcar.csv", 
                     col_types = cols(
                       age = col_double(),
                       education_num = col_double(),
                       hours_per_week = col_double(),
                       sex = col_factor(),
                       marital_status = col_factor(),
                       workclass = col_factor(),
                       occupation = col_factor(),
                       income = col_factor()
                     ),
                     show_col_types = FALSE)

mnar_data <- read_csv("adult_sample_mnar.csv", 
                     col_types = cols(
                       age = col_double(),
                       education_num = col_double(),
                       hours_per_week = col_double(),
                       sex = col_factor(),
                       marital_status = col_factor(),
                       workclass = col_factor(),
                       occupation = col_factor(),
                       income = col_factor()
                     ),
                     show_col_types = FALSE)

cat("Missing values in original datasets:\n")
cat("- MCAR:", sum(is.na(mcar_data)), "missing values\n")
cat("- MNAR:", sum(is.na(mnar_data)), "missing values\n\n")

# ----------------------------
# 2. Imputation Functions
# ----------------------------

# ----- MICE Imputation -----
perform_mice_imputation <- function(data, dataset_name, m = 5, maxit = 10) {
  cat("\n=== MICE Imputation for", dataset_name, "===\n")
  
  # Ensure factors are properly set
  data <- data %>% mutate(across(where(is.character), as.factor))
  
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
          methods[col] <- "logreg"  # Logistic regression for binary (from Google)
        } else {
          methods[col] <- "polyreg"  # Polytomous regression for categorical (from Google)
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

# ----- FAMD Imputation -----
perform_famd_imputation <- function(data, dataset_name, ncp = 2) {
  cat("\n=== FAMD Imputation for", dataset_name, "===\n")
  
  # Ensure factors are properly set
  data <- data %>% mutate(across(where(is.character), as.factor))
  
  # Show data types
  cat("Data types:\n")
  print(sapply(data, class))
  
  # Perform FAMD imputation (single imputation)
  cat("Performing FAMD imputation (ncp =", ncp, ")...\n")
  imputed_data <- missMDA::imputeFAMD(
    X = data,
    ncp = ncp,
    method = "Regularized",
    seed = 42
  )$completeObs
  
  # Convert to data frame with proper types
  imputed_df <- as.data.frame(imputed_data)
  for (col in names(imputed_df)) {
    if (is.factor(data[[col]])) {
      imputed_df[[col]] <- as.factor(imputed_df[[col]])
    }
  }
  
  return(imputed_df)
}

# ----- missForest Imputation (Revised) -----
perform_missforest_imputation <- function(data, dataset_name, maxiter = 10, ntree = 100) {
  cat("\n=== missForest Imputation for", dataset_name, "===\n")
  
  # Convert to data frame (missForest requires data frame)
  data_df <- as.data.frame(data)
  
  # Convert characters to factors
  data_df <- data_df %>% mutate(across(where(is.character), as.factor))
  
  # Show data types
  cat("Data types:\n")
  print(sapply(data_df, class))
  
  # Perform missForest
  cat("Performing missForest...\n")
  forest_result <- missForest::missForest(
    xmis = data_df,
    maxiter = maxiter,
    ntree = ntree,
    verbose = TRUE
  )
  
  # Return imputed data
  forest_result$ximp
}

### Ignore this, it doesn't work

# ----- rMIDAS Imputation -----
perform_rmidas_imputation <- function(data, dataset_name, training_epochs = 20, layer_structure = c(128, 128)) {
  cat("\n=== rMIDAS Imputation for", dataset_name, "===\n")
  
  # Convert to data frame
  data_df <- as.data.frame(data)
  
  # Ensure factors are properly set
  data_df <- data_df %>% mutate(across(where(is.character), as.factor))
  
  # Identify variable types
  factor_cols <- names(data_df)[sapply(data_df, is.factor)]
  binary_vars <- factor_cols[sapply(data_df[factor_cols], function(x) nlevels(x) == 2)]
  categorical_vars <- factor_cols[sapply(data_df[factor_cols], function(x) nlevels(x) > 2)]
  
  cat("Binary variables:", paste(binary_vars, collapse = ", "), "\n")
  cat("Categorical variables:", paste(categorical_vars, collapse = ", "), "\n")
  
  # Convert data for MIDAS
  cat("Converting data for rMIDAS...\n")
  data_converted <- rMIDAS::convert(
    data_df,
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
  "FAMD" = perform_famd_imputation,
  "missForest" = perform_missforest_imputation,
  "rMIDAS" = perform_rmidas_imputation
)

# Run all imputations
cat("\n\n", rep("=", 50), "\n")
cat("STARTING ALL IMPUTATIONS\n")
cat(rep("=", 50), "\n\n")

# For MCAR dataset
cat("\n", rep("-", 30), "\n")
cat("PROCESSING MCAR DATASET\n")
cat(rep("-", 30), "\n")
for (method_name in names(methods)) {
  cat("\nRunning", method_name, "on MCAR data...\n")
  result <- tryCatch({
    imputed <- methods[[method_name]](mcar_data, "MCAR")
    filename <- paste0("adult_sample_mcar_", method_name, ".csv")
    write_csv(imputed, filename)
    cat("Saved:", filename, "\n")
    na_count <- sum(is.na(imputed))
    cat(method_name, "MCAR missing values after imputation:", na_count, "\n")
    na_count
  }, error = function(e) {
    cat("!! ERROR: ", method_name, " failed on MCAR - ", e$message, "\n")
    -1  # Error indicator
  })
  
  if (result == -1) {
    cat("**", method_name, "imputation failed for MCAR\n")
  }
}

# For MNAR dataset
cat("\n", rep("-", 30), "\n")
cat("PROCESSING MNAR DATASET\n")
cat(rep("-", 30), "\n")
for (method_name in names(methods)) {
  cat("\nRunning", method_name, "on MNAR data...\n")
  result <- tryCatch({
    imputed <- methods[[method_name]](mnar_data, "MNAR")
    filename <- paste0("adult_sample_mnar_", method_name, ".csv")
    write_csv(imputed, filename)
    cat("Saved:", filename, "\n")
    na_count <- sum(is.na(imputed))
    cat(method_name, "MNAR missing values after imputation:", na_count, "\n")
    na_count
  }, error = function(e) {
    cat("!! ERROR: ", method_name, " failed on MNAR - ", e$message, "\n")
    -1  # Error indicator
  })
  
  if (result == -1) {
    cat("**", method_name, "imputation failed for MNAR\n")
  }
}

cat("\n\n=== SCRIPT COMPLETE ===")
cat("\nCheck working directory for output files\n")

# Shall we split these methods up into separate files for clarity and computational efficiency?
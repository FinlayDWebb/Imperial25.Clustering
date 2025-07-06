# =============================================
# DIBmix Imputation Evaluation Framework
# Optimized for DIBmix Clustering - FIXED VERSION
# Metrics: RMSE, PFC, and ARI
# =============================================

### 1. PACKAGE SETUP ###
# ----------------------
# Install necessary packages
required_packages <- c("tidyverse", "missForest", "mice", "aricode", "caret", "gridExtra")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  cat("Installing packages:", paste(new_packages, collapse = ", "), "\n")
  install.packages(new_packages, dependencies = TRUE)
}

# Load packages with error handling
suppressMessages({
  library(tidyverse)
  library(missForest)
  library(mice)
  library(aricode)
  library(caret)
  library(gridExtra)
})

# Try to install DIBclust with better error handling
install_dibclust <- function() {
  if (!require("DIBclust", quietly = TRUE)) {
    cat("Installing DIBclust from GitHub...\n")
    if (!require("devtools", quietly = TRUE)) {
      install.packages("devtools")
      library(devtools)
    }
    tryCatch({
      devtools::install_github("amarkos/DIBclust", quiet = TRUE)
      library(DIBclust)
      return(TRUE)
    }, error = function(e) {
      cat("Failed to install DIBclust:", e$message, "\n")
      cat("Continuing without DIBclust - will use alternative clustering\n")
      return(FALSE)
    })
  }
  return(TRUE)
}

dibclust_available <- install_dibclust()

### 2. DATA PREPARATION - SYNTHETIC DATA ###
# ------------------------------------------
# Create synthetic mixed-type data instead of downloading
set.seed(42)
create_synthetic_data <- function(n = 200) {
  # Create two distinct clusters
  cluster1 <- tibble(
    age = rnorm(n/2, mean = 35, sd = 8),
    income = rnorm(n/2, mean = 45000, sd = 10000),
    education = sample(c("High School", "Bachelor", "Master"), n/2, replace = TRUE, prob = c(0.6, 0.3, 0.1)),
    occupation = sample(c("Service", "Sales", "Technical"), n/2, replace = TRUE, prob = c(0.5, 0.3, 0.2)),
    true_cluster = 1
  )
  
  cluster2 <- tibble(
    age = rnorm(n/2, mean = 45, sd = 10),
    income = rnorm(n/2, mean = 75000, sd = 15000),
    education = sample(c("High School", "Bachelor", "Master"), n/2, replace = TRUE, prob = c(0.2, 0.4, 0.4)),
    occupation = sample(c("Service", "Sales", "Technical"), n/2, replace = TRUE, prob = c(0.2, 0.3, 0.5)),
    true_cluster = 2
  )
  
  # Combine and shuffle
  data <- bind_rows(cluster1, cluster2) %>%
    mutate(
      age = pmax(18, pmin(65, age)),  # Bound age
      income = pmax(20000, income),   # Bound income
      education = factor(education, levels = c("High School", "Bachelor", "Master")),
      occupation = factor(occupation, levels = c("Service", "Sales", "Technical"))
    ) %>%
    slice_sample(n = n)
  
  return(data)
}

# Generate synthetic data
adult <- create_synthetic_data(200)
cat("Generated synthetic data with", nrow(adult), "observations\n")

### 3. MISSING DATA SIMULATION ###
# -------------------------------
create_missing_data <- function(data, p) {
  # Create missing data pattern (MAR - Missing at Random)
  missing_mask <- matrix(
    runif(nrow(data) * ncol(data)) < p,
    nrow = nrow(data),
    ncol = ncol(data)
  )
  
  data_with_missing <- data
  data_with_missing[missing_mask] <- NA
  return(data_with_missing)
}

# Create missing data on the features we'll impute
features_to_impute <- c("age", "income", "education", "occupation")
adult_missing <- create_missing_data(adult[features_to_impute], 0.15)  # Reduced missing rate

# Identify categorical and continuous columns
cat_cols <- which(sapply(adult_missing, is.factor))
cont_cols <- which(sapply(adult_missing, is.numeric))

cat("Categorical columns:", cat_cols, "\n")
cat("Continuous columns:", cont_cols, "\n")

### 4. IMPUTATION METHODS ###
# --------------------------
# 1. Mean/Mode Imputation
mean_mode_impute <- function(data) {
  result <- data
  
  # Impute continuous variables with mean
  for (col in names(data)[cont_cols]) {
    if (is.numeric(data[[col]])) {
      result[[col]][is.na(result[[col]])] <- mean(data[[col]], na.rm = TRUE)
    }
  }
  
  # Impute categorical variables with mode
  for (col in names(data)[cat_cols]) {
    if (is.factor(data[[col]])) {
      mode_val <- names(which.max(table(data[[col]])))
      result[[col]][is.na(result[[col]])] <- mode_val
      result[[col]] <- factor(result[[col]], levels = levels(data[[col]]))
    }
  }
  
  return(result)
}

# 2. MICE Imputation with reduced iterations
mice_impute <- function(data) {
  suppressMessages({
    mice_result <- mice(data, m = 1, maxit = 5, printFlag = FALSE)
    complete(mice_result)
  })
}

# 3. missForest Imputation with reduced iterations
missforest_impute <- function(data) {
  suppressMessages({
    result <- missForest(data, maxiter = 5, ntree = 10)
    result$ximp
  })
}

### 5. EVALUATION METRICS ###
# --------------------------
# RMSE for continuous variables
calculate_rmse <- function(original, imputed) {
  if (all(is.na(original)) || all(is.na(imputed))) return(NA)
  sqrt(mean((original - imputed)^2, na.rm = TRUE))
}

# PFC for categorical variables
calculate_pfc <- function(original, imputed) {
  if (all(is.na(original)) || all(is.na(imputed))) return(NA)
  mean(original != imputed, na.rm = TRUE)
}

# Combined evaluation function
evaluate_imputation <- function(original, imputed) {
  results <- list()
  
  # RMSE for continuous variables
  for (col in names(original)[cont_cols]) {
    if (is.numeric(original[[col]]) && is.numeric(imputed[[col]])) {
      results[[paste0("RMSE_", col)]] <- calculate_rmse(original[[col]], imputed[[col]])
    }
  }
  
  # PFC for categorical variables
  for (col in names(original)[cat_cols]) {
    if (is.factor(original[[col]]) && is.factor(imputed[[col]])) {
      results[[paste0("PFC_", col)]] <- calculate_pfc(original[[col]], imputed[[col]])
    }
  }
  
  # Overall metrics
  rmse_values <- results[grepl("RMSE", names(results))]
  pfc_values <- results[grepl("PFC", names(results))]
  
  results$Overall_RMSE <- mean(unlist(rmse_values), na.rm = TRUE)
  results$Overall_PFC <- mean(unlist(pfc_values), na.rm = TRUE)
  
  return(results)
}

### 6. CLUSTERING METHODS ###
# --------------------------
# Alternative clustering if DIBclust is not available
run_kmeans_clustering <- function(data) {
  # Convert factors to numeric for k-means
  data_numeric <- data
  for (col in names(data)) {
    if (is.factor(data[[col]])) {
      data_numeric[[col]] <- as.numeric(data[[col]])
    }
  }
  
  # Scale the data
  data_scaled <- scale(data_numeric)
  
  # Run k-means
  kmeans_result <- kmeans(data_scaled, centers = 2, nstart = 10)
  return(kmeans_result$cluster)
}

# DIBmix clustering (if available)
run_dibmix_clustering <- function(data) {
  if (!dibclust_available) {
    return(run_kmeans_clustering(data))
  }
  
  tryCatch({
    # Prepare data for DIBmix
    scaled_data <- data
    if (length(cont_cols) > 0) {
      scaled_data[, cont_cols] <- scale(data[, cont_cols])
    }
    
    # Run DIBmix with conservative parameters
    dib_result <- DIBmix(
      X = scaled_data,
      ncl = 2,
      catcols = cat_cols,
      contcols = cont_cols,
      randinit = NULL,
      s = -1,
      lambda = -1,
      scale = FALSE,
      maxiter = 50,  # Reduced iterations
      nstart = 10    # Reduced starts
    )
    
    return(dib_result$Cluster)
  }, error = function(e) {
    cat("DIBmix failed, using k-means:", e$message, "\n")
    return(run_kmeans_clustering(data))
  })
}

### 7. MAIN EVALUATION ###
# ------------------------
# Define methods to test
imputation_methods <- list(
  "Mean/Mode" = mean_mode_impute,
  "MICE" = mice_impute,
  "missForest" = missforest_impute
)

# Initialize results storage
results <- tibble()

# Ground truth (original data without missing values)
ground_truth <- adult[features_to_impute]
true_clusters <- adult$true_cluster

cat("\n=== STARTING EVALUATION ===\n")
cat("Ground truth clusters:", table(true_clusters), "\n")

for (method_name in names(imputation_methods)) {
  cat("\nProcessing", method_name, "...\n")
  
  # Apply imputation
  start_time <- Sys.time()
  imputed_data <- tryCatch({
    imputation_methods[[method_name]](adult_missing)
  }, error = function(e) {
    cat("Imputation failed for", method_name, ":", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(imputed_data)) {
    cat("Skipping", method_name, "due to imputation failure\n")
    next
  }
  
  end_time <- Sys.time()
  cat("Imputation completed in", round(difftime(end_time, start_time, units = "secs"), 2), "seconds\n")
  
  # Calculate metrics
  metrics <- evaluate_imputation(ground_truth, imputed_data)
  
  # Cluster with DIBmix or alternative
  cat("Running clustering...\n")
  clusters <- tryCatch({
    run_dibmix_clustering(imputed_data)
  }, error = function(e) {
    cat("Clustering failed for", method_name, ":", e$message, "\n")
    rep(NA, nrow(imputed_data))
  })
  
  # Calculate ARI if clustering succeeded
  ari_value <- if (all(is.na(clusters))) {
    NA
  } else {
    cat("Predicted clusters:", table(clusters), "\n")
    ARI(true_clusters, clusters)
  }
  
  # Store results
  result_row <- tibble(
    Method = method_name,
    Overall_RMSE = metrics$Overall_RMSE,
    Overall_PFC = metrics$Overall_PFC,
    ARI = ari_value
  )
  
  # Add individual metrics
  for (metric_name in names(metrics)) {
    if (!metric_name %in% c("Overall_RMSE", "Overall_PFC")) {
      result_row[[metric_name]] <- metrics[[metric_name]]
    }
  }
  
  results <- bind_rows(results, result_row)
  cat("Completed", method_name, "- ARI:", round(ari_value, 3), "\n")
}

### 8. VISUALIZATION ###
# ---------------------
if (nrow(results) > 0) {
  # RMSE vs ARI plot
  plot_rmse_ari <- ggplot(results, aes(x = Overall_RMSE, y = ARI, color = Method)) +
    geom_point(size = 4) +
    geom_text(aes(label = Method), vjust = -0.5, size = 3) +
    labs(title = "Reconstruction Error vs Cluster Preservation",
         x = "Overall RMSE (Lower is Better)",
         y = "ARI (Higher is Better)") +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  # PFC comparison plot
  plot_pfc <- ggplot(results, aes(x = Method, y = Overall_PFC, fill = Method)) +
    geom_bar(stat = "identity") +
    labs(title = "Overall Categorical Imputation Error",
         y = "Overall PFC (Lower is Better)") +
    theme_minimal() +
    theme(legend.position = "none")
  
  # Display plots
  print(plot_rmse_ari)
  print(plot_pfc)
  
  # Try to arrange plots if possible
  tryCatch({
    grid.arrange(plot_rmse_ari, plot_pfc, ncol = 2)
  }, error = function(e) {
    cat("Could not arrange plots:", e$message, "\n")
  })
}

### 9. RESULTS OUTPUT ###
# ----------------------
cat("\n=== FINAL RESULTS ===\n")
if (nrow(results) > 0) {
  print(results)
  
  # Identify brittle methods
  if (nrow(results) > 1) {
    brittle_methods <- results %>%
      filter(Overall_RMSE < median(Overall_RMSE, na.rm = TRUE) & 
             Overall_PFC > median(Overall_PFC, na.rm = TRUE))
    
    if (nrow(brittle_methods) > 0) {
      cat("\nWARNING: Brittle methods detected (good RMSE but poor PFC):\n")
      print(brittle_methods)
    } else {
      cat("\nNo brittle methods detected.\n")
    }
  }
  
  # Save results
  tryCatch({
    write_csv(results, "dibmix_imputation_evaluation_results.csv")
    cat("\nResults saved to 'dibmix_imputation_evaluation_results.csv'\n")
  }, error = function(e) {
    cat("Could not save results:", e$message, "\n")
  })
} else {
  cat("No results to display - all methods failed\n")
}

cat("\n=== EVALUATION COMPLETE ===\n")
### Here the outline is to do the following:
# 1. Load the data
# 2. Cluster the original dataset
# 3. Cluster the imputed datasets
# 4. Compare the ARI for each

# This works, because instead of measuring the imputed data against the datasets ground truths (i.e. categories)
# We instead test it relatively against the clustererd data before imputation, that becomes our ground truth.
# This means that how well imputation preserves the natural structure of the data, rather than, how well the imputation
# preserves the supervised learning potential of the data. 

# We want to know strenghts/weaknesses with general datatypes, non-linearity, homogeneity, etc. Not how well imputation
# preserves the ability to predict a target variable (like income, or age, etc.).

# Load necessary libraries

# Do this with DIBmix, and note we have to fix the bandwidths in DIBmix. But let us first find a good bandwidth that works.

# Clustering and ARI Analysis for Imputed Datasets
# Comparing clustering structure preservation across different imputation methods

# Load necessary libraries
library(IBclust)
library(cluster)
library(dplyr)
library(readr)

# Function to calculate ARI (Adjusted Rand Index)
calculate_ari <- function(cluster1, cluster2) {
  # Convert to factors if needed
  cluster1 <- as.factor(cluster1)
  cluster2 <- as.factor(cluster2)
  
  # Calculate ARI using cluster package
  ari <- cluster::adjustedRandIndex(cluster1, cluster2)
  return(ari)
}

# Function to identify variable types automatically
identify_variable_types <- function(data) {
  categorical_vars <- c()
  continuous_vars <- c()
  
  for (col in names(data)) {
    if (is.factor(data[[col]]) || is.character(data[[col]])) {
      categorical_vars <- c(categorical_vars, col)
    } else if (is.numeric(data[[col]])) {
      # Check if it's binary (0/1) or has few unique values
      unique_vals <- length(unique(data[[col]][!is.na(data[[col]])]))
      if (unique_vals <= 10) {
        # Convert to factor for clustering
        data[[col]] <- as.factor(data[[col]])
        categorical_vars <- c(categorical_vars, col)
      } else {
        continuous_vars <- c(continuous_vars, col)
      }
    }
  }
  
  return(list(
    data = data,
    categorical_vars = categorical_vars,
    continuous_vars = continuous_vars
  ))
}

# Function to perform clustering using DIBmix
perform_clustering <- function(data, ncl = 3, nstart = 50) {
  # Identify variable types
  var_info <- identify_variable_types(data)
  data_processed <- var_info$data
  cat_vars <- var_info$categorical_vars
  cont_vars <- var_info$continuous_vars
  
  cat("Categorical variables:", paste(cat_vars, collapse = ", "), "\n")
  cat("Continuous variables:", paste(cont_vars, collapse = ", "), "\n")
  
  # Get column indices
  cat_cols <- which(names(data_processed) %in% cat_vars)
  cont_cols <- which(names(data_processed) %in% cont_vars)
  
  # Perform clustering based on data type composition
  if (length(cat_cols) > 0 && length(cont_cols) > 0) {
    # Mixed-type clustering
    cat("Using DIBmix for mixed-type data\n")
    result <- DIBmix(X = data_processed, ncl = ncl, 
                     catcols = cat_cols, contcols = cont_cols,
                     nstart = nstart)
  } else if (length(cat_cols) > 0) {
    # Categorical-only clustering
    cat("Using DIBcat for categorical data\n")
    result <- DIBcat(X = data_processed, ncl = ncl, 
                     lambda = -1, nstart = nstart)
  } else if (length(cont_cols) > 0) {
    # Continuous-only clustering
    cat("Using DIBcont for continuous data\n")
    result <- DIBcont(X = as.matrix(data_processed), ncl = ncl, 
                      s = -1, nstart = nstart)
  } else {
    stop("No valid variables found for clustering")
  }
  
  return(result)
}

# Function to load and process a single dataset
load_and_cluster <- function(filename, ncl = 3) {
  cat("\n=== Processing", filename, "===\n")
  
  # Check if file exists
  if (!file.exists(filename)) {
    cat("File not found:", filename, "\n")
    return(NULL)
  }
  
  # Load data
  data <- read_csv(filename, show_col_types = FALSE)
  cat("Data dimensions:", nrow(data), "x", ncol(data), "\n")
  
  # Remove any ID columns or irrelevant columns
  if ("Unnamed: 0" %in% names(data)) {
    data <- data %>% select(-`Unnamed: 0`)
  }
  if ("X" %in% names(data)) {
    data <- data %>% select(-X)
  }
  
  # Check for missing values
  missing_count <- sum(is.na(data))
  cat("Missing values:", missing_count, "\n")
  
  # Perform clustering
  tryCatch({
    result <- perform_clustering(data, ncl = ncl)
    cat("Clustering completed successfully\n")
    cat("Entropy:", result$Entropy, "\n")
    cat("Mutual Information:", result$MutualInfo, "\n")
    
    return(list(
      data = data,
      clusters = result$Cluster,
      entropy = result$Entropy,
      mutual_info = result$MutualInfo,
      filename = filename
    ))
  }, error = function(e) {
    cat("Error in clustering:", e$message, "\n")
    return(NULL)
  })
}

# Function to find optimal number of clusters
find_optimal_clusters <- function(data, max_k = 6) {
  results <- list()
  
  for (k in 2:max_k) {
    cat("Testing k =", k, "\n")
    tryCatch({
      result <- perform_clustering(data, ncl = k)
      results[[paste0("k", k)]] <- list(
        k = k,
        entropy = result$Entropy,
        mutual_info = result$MutualInfo,
        clusters = result$Cluster
      )
    }, error = function(e) {
      cat("Error for k =", k, ":", e$message, "\n")
    })
  }
  
  return(results)
}

# Main analysis function
main_analysis <- function() {
  # Define dataset files
  datasets <- c(
    "adult_sample_mcar_FAMD.csv",
    "adult_sample_mnar_FAMD.csv",
    "adult_sample_mcar_MICE.csv",
    "adult_sample_mnar_MICE.csv",
    "adult_sample_mcar_MIDAS.csv",
    "adult_sample_mnar_MIDAS.csv",
    "adult_sample_mcar_missForest.csv",
    "adult_sample_mnar_missForest.csv"
  )
  
  # Load original datasets (assuming they exist)
  original_mcar <- "adult_sample_mcar.csv"  # Original with MCAR missingness
  original_mnar <- "adult_sample_mnar.csv"  # Original with MNAR missingness
  
  # Initialize results storage
  clustering_results <- list()
  ari_results <- data.frame(
    Dataset = character(),
    Method = character(),
    Missingness_Type = character(),
    ARI_vs_Original = numeric(),
    Entropy = numeric(),
    Mutual_Info = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Process each dataset
  for (dataset in datasets) {
    result <- load_and_cluster(dataset, ncl = 3)
    if (!is.null(result)) {
      clustering_results[[dataset]] <- result
      
      # Extract method and missingness type from filename
      method <- sub(".*_([A-Za-z]+)\\.csv$", "\\1", dataset)
      miss_type <- ifelse(grepl("mcar", dataset), "MCAR", "MNAR")
      
      # Store results
      ari_results <- rbind(ari_results, data.frame(
        Dataset = dataset,
        Method = method,
        Missingness_Type = miss_type,
        ARI_vs_Original = NA,  # Will calculate below
        Entropy = result$entropy,
        Mutual_Info = result$mutual_info,
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # Load and cluster original datasets (if available)
  original_results <- list()
  if (file.exists(original_mcar)) {
    original_results$mcar <- load_and_cluster(original_mcar, ncl = 3)
  }
  if (file.exists(original_mnar)) {
    original_results$mnar <- load_and_cluster(original_mnar, ncl = 3)
  }
  
  # Calculate ARI values comparing imputed vs original
  for (i in 1:nrow(ari_results)) {
    dataset_name <- ari_results$Dataset[i]
    miss_type <- tolower(ari_results$Missingness_Type[i])
    
    if (dataset_name %in% names(clustering_results) && 
        miss_type %in% names(original_results)) {
      
      imputed_clusters <- clustering_results[[dataset_name]]$clusters
      original_clusters <- original_results[[miss_type]]$clusters
      
      # Calculate ARI
      ari_value <- calculate_ari(imputed_clusters, original_clusters)
      ari_results$ARI_vs_Original[i] <- ari_value
    }
  }
  
  # Print results
  cat("\n=== CLUSTERING ANALYSIS RESULTS ===\n")
  print(ari_results)
  
  # Summary statistics by method
  cat("\n=== SUMMARY BY METHOD ===\n")
  summary_by_method <- ari_results %>%
    group_by(Method) %>%
    summarise(
      Mean_ARI = mean(ARI_vs_Original, na.rm = TRUE),
      SD_ARI = sd(ARI_vs_Original, na.rm = TRUE),
      Mean_Entropy = mean(Entropy, na.rm = TRUE),
      Mean_MutualInfo = mean(Mutual_Info, na.rm = TRUE),
      .groups = 'drop'
    )
  print(summary_by_method)
  
  # Summary statistics by missingness type
  cat("\n=== SUMMARY BY MISSINGNESS TYPE ===\n")
  summary_by_miss <- ari_results %>%
    group_by(Missingness_Type) %>%
    summarise(
      Mean_ARI = mean(ARI_vs_Original, na.rm = TRUE),
      SD_ARI = sd(ARI_vs_Original, na.rm = TRUE),
      Mean_Entropy = mean(Entropy, na.rm = TRUE),
      Mean_MutualInfo = mean(Mutual_Info, na.rm = TRUE),
      .groups = 'drop'
    )
  print(summary_by_miss)
  
  # Best performing method
  cat("\n=== BEST PERFORMING METHOD ===\n")
  best_method <- ari_results %>%
    filter(!is.na(ARI_vs_Original)) %>%
    arrange(desc(ARI_vs_Original)) %>%
    head(1)
  print(best_method)
  
  return(list(
    clustering_results = clustering_results,
    ari_results = ari_results,
    original_results = original_results,
    summary_by_method = summary_by_method,
    summary_by_miss = summary_by_miss
  ))
}

# Run the analysis
cat("Starting clustering analysis...\n")
results <- main_analysis()

# Save results
write.csv(results$ari_results, "clustering_ari_results.csv", row.names = FALSE)
cat("\nResults saved to 'clustering_ari_results.csv'\n")

# Additional analysis: Pairwise ARI between methods
cat("\n=== PAIRWISE ARI BETWEEN METHODS ===\n")
pairwise_ari <- function(clustering_results) {
  methods <- names(clustering_results)
  n_methods <- length(methods)
  
  if (n_methods < 2) {
    cat("Not enough methods for pairwise comparison\n")
    return(NULL)
  }
  
  pairwise_results <- data.frame(
    Method1 = character(),
    Method2 = character(),
    ARI = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (i in 1:(n_methods-1)) {
    for (j in (i+1):n_methods) {
      method1 <- methods[i]
      method2 <- methods[j]
      
      clusters1 <- clustering_results[[method1]]$clusters
      clusters2 <- clustering_results[[method2]]$clusters
      
      if (length(clusters1) == length(clusters2)) {
        ari_value <- calculate_ari(clusters1, clusters2)
        pairwise_results <- rbind(pairwise_results, data.frame(
          Method1 = method1,
          Method2 = method2,
          ARI = ari_value,
          stringsAsFactors = FALSE
        ))
      }
    }
  }
  
  return(pairwise_results)
}

# Calculate pairwise ARI
pairwise_results <- pairwise_ari(results$clustering_results)
if (!is.null(pairwise_results)) {
  print(pairwise_results)
  write.csv(pairwise_results, "pairwise_ari_results.csv", row.names = FALSE)
}

cat("\nAnalysis completed!\n")
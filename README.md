# Imperial25.Clustering
TBD

### Left to do

1. Run the pipeline with 3-5 datasets, and see what the Lambda and Bandwidths on DIBmix return with. Then set a value accordingly. (This will have to be done by Efthymios).
2. Run the investigation with all preferred 12 real life datasets.
3. Create visualisations for the results and the poster and handle results.
4. Write up report, design poster. 
    On the report, datasets section nearly done.
    Must also add, restraints and problems that we adapted around (categorical, DIB bug, auto categorical detector?)
    Then results and conclusion, this goes hand-in-hand with 3.
    Oh, and the bibliography.


- I have the datasets, just need to do metadata, and remove automatic type checking.
- I've done the metadata, now its just time to tune the scripts so that I can send it off to Efthymios, I have to double check that the MICE missingness fix (I.e. after imputation, using mode/mean) is a valid method. It might ruin RMSE.


------

Spare Identify Columns Function for Cluster Pipe:

"""

identify_column_types <- function(data, metadata_path) {
  #' Identify column types USING METADATA
  #' 
  #' @param data Dataframe to analyze
  #' @param metadata_path Path to metadata CSV
  #' @return List with categorical and continuous column indices
  
  # Read metadata
  metadata <- read_csv(metadata_path, show_col_types = FALSE)

  # Debug printouts
  cat("Metadata columns:", names(metadata), "\n")
  if (nrow(metadata) > 0) {
    cat("First row values:\n")
    print(metadata[1, ])  # Use print() for data frames
  }

  cat_cols <- c()
  cont_cols <- c()
  
  for (i in 1:ncol(data)) {
    col_name <- names(data)[i]
    meta <- metadata[metadata$variable == col_name, ]
    
    if (nrow(meta) == 0) {
      cat(sprintf("Warning: No metadata for column '%s'. Skipping.\n", col_name))
      next
    }
    
    if (meta$type %in% c("categorical", "ordered")) {
      cat_cols <- c(cat_cols, i)
      cat(sprintf("Column %d (%s): Categorical (from metadata)\n", 
                 i, col_name))
    } 
    else if (meta$type == "numeric") {
      cont_cols <- c(cont_cols, i)
      cat(sprintf("Column %d (%s): Continuous (from metadata)\n", 
                 i, col_name))
    }
  }
  
  return(list(
    data = data,
    categorical = cat_cols,
    continuous = cont_cols
  ))
}

"""

------

And for perform DIBmix:

"""

perform_dibmix_clustering <- function(data, n_clusters = 2, metadata_path) {
  #' Perform DIBmix clustering on mixed-type data
  
  tryCatch({
    # ENSURE DATA IS A PROPER DATA.FRAME
    data <- as.data.frame(data)
    
    # DEBUG: Check data types at entry to clustering
    cat("=== CLUSTERING DEBUG ===\n")
    cat("Data class at clustering entry:", class(data), "\n")
    cat("Data classes in clustering function:\n")
    for (col in names(data)[1:min(10, ncol(data))]) {
      cat(sprintf("%s: %s (is.factor: %s, is.ordered: %s, is.numeric: %s)\n", 
                  col, 
                  paste(class(data[[col]]), collapse = ", "),
                  is.factor(data[[col]]),
                  is.ordered(data[[col]]),
                  is.numeric(data[[col]])))
    }
    
    # More robust factor detection
    cat_cols <- c()
    cont_cols <- c()
    
    for (col in names(data)) {
      col_data <- data[[col]]
      if ("factor" %in% class(col_data) || "ordered" %in% class(col_data)) {
        cat_cols <- c(cat_cols, col)
      } else if (is.numeric(col_data)) {
        cont_cols <- c(cont_cols, col)
      }
    }
    
    cat("Factor detection results:\n")
    cat("cat_cols found:", length(cat_cols), "->", paste(cat_cols[1:min(5, length(cat_cols))], collapse=", "), "\n")
    cat("cont_cols found:", length(cont_cols), "->", paste(cont_cols[1:min(5, length(cont_cols))], collapse=", "), "\n")
    
    # Clean data for DIBmix
    if (length(cont_cols) > 0) {
      for (col in cont_cols) {
        data[[col]][is.infinite(data[[col]])] <- NA
      }
      data <- data[complete.cases(data[, cont_cols, drop = FALSE]), ]
    }
    
    if (length(cat_cols) > 0) {
      for (col in cat_cols) {
        if (is.factor(data[[col]])) {
          data[[col]] <- droplevels(data[[col]])
        }
        if (any(is.na(data[[col]]))) {
          cat(sprintf("WARNING: NAs found in factor column %s\n", col))
          data <- data[!is.na(data[[col]]), ]
        }
      }
    }
    
    cat(sprintf("DIBmix clustering: %d categorical, %d continuous columns\n",
                length(cat_cols), length(cont_cols)))
    
    cat("Final data dimensions:", dim(data), "\n")
    
    # Perform clustering based on data types
    if (length(cat_cols) > 0 && length(cont_cols) > 0) {
      cat("Using DIBmix for mixed data\n")
      result <- DIBmix(X = data,
                        ncl = n_clusters,
                        catcols = which(names(data) %in% cat_cols),
                        contcols = which(names(data) %in% cont_cols),
                        s = -1,
                        lambda = -1,
                        nstart = 50)
    } else if (length(cont_cols) > 0) {
      cat("Using DIBcont for continuous data\n")
      X_cont <- as.matrix(data[, cont_cols, drop = FALSE])
      result <- DIBcont(X = X_cont, ncl = n_clusters, s = -1, nstart = 50)
    } else if (length(cat_cols) > 0) {
      cat("Using DIBcat for categorical data\n")
      X_cat <- data[, cat_cols, drop = FALSE]
      result <- DIBcat(X = X_cat, ncl = n_clusters, lambda = -1, nstart = 50)
    } else {
      stop("No valid columns found for clustering")
    }
    
    if (is.null(result) || !"Cluster" %in% names(result)) {
      stop("Clustering returned invalid result object")
    }
    
    result$categorical_cols <- cat_cols
    result$continuous_cols <- cont_cols
    result$n_clusters <- n_clusters
    
    cat(sprintf("Clustering completed: Entropy = %.4f, Mutual Info = %.4f\n",
                result$Entropy, result$MutualInfo))
    cat("Cluster assignments:", head(result$Cluster), "\n")
    
    return(result)
    
  }, error = function(e) {
    cat("\n!!! CLUSTERING FAILURE DETAILS !!!\n")
    cat("Error message:", e$message, "\n")
    cat("Data dimensions:", dim(data), "\n")
    if (exists("cat_cols")) {
      cat("Categorical columns found:", length(cat_cols), "\n")
      cat("Continuous columns found:", length(cont_cols), "\n")
    }
    cat("n_clusters:", n_clusters, "\n")
    
    return(NULL)
  })
}

"""
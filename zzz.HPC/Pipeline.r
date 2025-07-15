# master_pipeline.R
library(readr)
library(dplyr)
library(mice)
library(missMDA)
library(missForest)
library(doParallel)

# --------------------
# 1. Preprocessing
# --------------------
preprocess_data <- function(input_file, sample_size = 7500) {
  cat("Preprocessing data...\n")
  data <- read_csv(input_file, col_types = cols(.default = "c"))
  
  # Standard column names
  col_names <- c("age", "workclass", "fnlwgt", "education", "education_num", 
                 "marital_status", "occupation", "relationship", "race", "sex", 
                 "capital_gain", "capital_loss", "hours_per_week", "native_country", 
                 "income")
  colnames(data) <- col_names
  
  # Cleaning pipeline
  data_clean <- data %>%
    mutate(across(everything(), ~trimws(as.character(.)))) %>%
    mutate(across(everything(), ~ifelse(. == "?", NA, .))) %>%
    mutate(
      age = as.numeric(age),
      education_num = as.numeric(education_num),
      hours_per_week = as.numeric(hours_per_week),
      capital_gain = as.numeric(capital_gain),
      capital_loss = as.numeric(capital_loss),
      fnlwgt = as.numeric(fnlwgt)
    ) %>%
    select(age, education_num, hours_per_week, sex, 
           marital_status, workclass, occupation, income) %>%
    na.omit() %>%
    slice_sample(n = min(sample_size, n()))
  
  write_csv(data_clean, "adult_sample_processed.csv")
  cat("Preprocessed data saved\n")
  return(data_clean)
}

# --------------------
# 2. Missingness Injection
# --------------------
inject_missingness <- function(data, pattern = "MCAR") {
  cat("Injecting", pattern, "missingness...\n")
  
  if (pattern == "MCAR") {
    data_missing <- data
    for(col in c("hours_per_week", "education_num")) {
      n_missing <- round(nrow(data) * 0.2)
      missing_idx <- sample(1:nrow(data), n_missing)
      data_missing[missing_idx, col] <- NA
    }
    write_csv(data_missing, paste0("adult_sample_", pattern, ".csv"))
  } 
  else if (pattern == "MNAR") {
    # Custom MNAR patterns
    data_missing <- data
    
    # Pattern 1: Income-based missingness for hours_per_week
    high_inc_idx <- which(data$income == ">50K")
    low_inc_idx <- which(data$income == "<=50K")
    missing_high <- sample(high_inc_idx, round(length(high_inc_idx) * 0.35))
    missing_low <- sample(low_inc_idx, round(length(low_inc_idx) * 0.10))
    data_missing[c(missing_high, missing_low), "hours_per_week"] <- NA
    
    # Pattern 2: Education-based missingness
    prob_missing <- ifelse(data$education_num <= 9, 0.40,
                          ifelse(data$education_num <= 12, 0.15, 0.05))
    missing_edu <- which(runif(nrow(data)) < prob_missing)
    data_missing[missing_edu, "education_num"] <- NA
    
    write_csv(data_missing, paste0("adult_sample_", pattern, ".csv"))
  }
  
  return(data_missing)
}

# --------------------
# 3. Imputation Functions
# --------------------
impute_mice <- function(data) {
  data <- data %>% mutate(across(where(is.character), as.factor))
  imp <- mice(data, m = 1, maxit = 10, seed = 42)
  complete(imp, 1)
}

impute_famd <- function(data) {
  data <- data %>% mutate(across(where(is.character), as.factor))
  imp <- imputeFAMD(data, ncp = 2)$completeObs
  imp %>% mutate(across(where(is.numeric), as.numeric))
}

impute_missforest <- function(data) {
  data <- data %>% mutate(across(where(is.character), as.factor))
  missForest(data)$ximp
}

# --------------------
# 4. Evaluation Metrics
# --------------------
calculate_metrics <- function(original, imputed) {
  metrics <- list()
  
  # RMSE for numeric
  num_cols <- names(original)[sapply(original, is.numeric)]
  if(length(num_cols) > 0) {
    rmse_vals <- sapply(num_cols, function(col) {
      sqrt(mean((original[[col]] - imputed[[col]])^2, na.rm = TRUE))
    })
    metrics$RMSE <- mean(rmse_vals)
  }
  
  # PFC for categorical
  cat_cols <- names(original)[sapply(original, is.character) | sapply(original, is.factor)]
  if(length(cat_cols) > 0) {
    pfc_vals <- sapply(cat_cols, function(col) {
      mean(original[[col]] != imputed[[col]], na.rm = TRUE)
    })
    metrics$PFC <- mean(pfc_vals)
  }
  
  # Correlation
  if(length(num_cols) > 0) {
    cor_vals <- sapply(num_cols, function(col) {
      cor(original[[col]], imputed[[col]], use = "complete.obs")
    })
    metrics$Correlation <- mean(cor_vals)
  }
  
  return(metrics)
}

# --------------------
# 5. Main Pipeline
# --------------------
run_pipeline <- function(dataset_path) {
  # Preprocessing
  clean_data <- preprocess_data(dataset_path)
  
  # Create missing datasets
  mcar_data <- inject_missingness(clean_data, "MCAR")
  mnar_data <- inject_missingness(clean_data, "MNAR")
  
  # Imputation methods
  methods <- list(
    "MICE" = impute_mice,
    "FAMD" = impute_famd,
    "missForest" = impute_missforest
  )
  
  # Results storage
  results <- data.frame()
  
  # Parallel imputation
  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  
  for(pattern in c("MCAR", "MNAR")) {
    data <- get(paste0(tolower(pattern), "_data"))
    
    foreach(method_name = names(methods), .combine = rbind) %dopar% {
      # Impute
      imputed <- methods[[method_name]](data)
      
      # Save
      filename <- paste0("adult_sample_", pattern, "_", method_name, ".csv")
      write_csv(imputed, filename)
      
      # Evaluate
      metrics <- calculate_metrics(clean_data, imputed)
      
      # Record results
      res <- data.frame(
        Dataset = basename(dataset_path),
        Method = method_name,
        Missing_Pattern = pattern,
        RMSE = metrics$RMSE %||% NA,
        PFC = metrics$PFC %||% NA,
        Correlation = metrics$Correlation %||% NA
      )
      
      # Append to global results
      results <<- rbind(results, res)
    }
  }
  
  stopCluster(cl)
  
  # Save final results
  write_csv(results, "r_imputation_results.csv")
  return(results)
}

# Execute pipeline
run_pipeline("adult.data")
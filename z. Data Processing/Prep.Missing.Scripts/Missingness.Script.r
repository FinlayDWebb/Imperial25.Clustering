library(readr)
library(dplyr)
library(utils)

# Create MCAR and MNAR Missing Data Script
# This script creates realistic missing data patterns for imputation evaluation

# Load the processed data
adult_complete <- read.csv("adult_sample_processed.csv", stringsAsFactors = FALSE)

cat("Original dataset dimensions:", dim(adult_complete), "\n")
cat("Original dataset summary:\n")
summary(adult_complete)

# Set seed for reproducibility
set.seed(123)

# ===============================
# MCAR - Missing Completely At Random
# ===============================

create_mcar_data <- function(data, missing_rate = 0.20) {
  adult_mcar <- data
  
  # Columns to introduce MCAR missingness
  mcar_columns <- c("hours_per_week", "education_num")
  
  cat("\n=== Creating MCAR Dataset ===\n")
  
  for(col in mcar_columns) {
    if(col %in% names(adult_mcar)) {
      n_rows <- nrow(adult_mcar)
      n_missing <- round(n_rows * missing_rate)
      
      # Completely random selection
      missing_indices <- sample(1:n_rows, n_missing)
      adult_mcar[missing_indices, col] <- NA
      
      cat(paste0("MCAR - Column '", col, "': ", n_missing, " missing values (", 
                round(missing_rate*100, 1), "%)\n"))
    }
  }
  
  return(adult_mcar)
}

# ===============================
# MNAR - Missing Not At Random
# ===============================

create_mnar_data <- function(data, missing_rate = 0.20) {
  adult_mnar <- data
  
  cat("\n=== Creating MNAR Dataset ===\n")
  
  # MNAR Pattern 1: Higher income people less likely to report hours_per_week
  # (Privacy concern - high earners might not want to reveal they work fewer hours)
  if("hours_per_week" %in% names(adult_mnar) && "income" %in% names(adult_mnar)) {
    high_income_indices <- which(adult_mnar$income == ">50K")
    n_high_income_missing <- round(length(high_income_indices) * 0.35)  # 35% of high earners
    
    low_income_indices <- which(adult_mnar$income == "<=50K")
    n_low_income_missing <- round(length(low_income_indices) * 0.10)   # 10% of low earners
    
    # Introduce missingness
    missing_high <- sample(high_income_indices, n_high_income_missing)
    missing_low <- sample(low_income_indices, n_low_income_missing)
    
    adult_mnar[c(missing_high, missing_low), "hours_per_week"] <- NA
    
    cat(paste0("MNAR - hours_per_week: ", n_high_income_missing, " missing from high income, ",
              n_low_income_missing, " missing from low income\n"))
  }
  
  # MNAR Pattern 2: People with lower education less likely to report education_num
  # (Social desirability bias - people with less education might skip this question)
  if("education_num" %in% names(adult_mnar)) {
    # Higher probability of missing for lower education levels
    prob_missing <- ifelse(adult_mnar$education_num <= 9, 0.40,    # 40% for <= 9 years
                          ifelse(adult_mnar$education_num <= 12, 0.15, # 15% for 10-12 years
                                0.05))  # 5% for > 12 years
    
    # Generate random numbers and compare to probability
    random_vals <- runif(nrow(adult_mnar))
    missing_indices <- which(random_vals < prob_missing)
    
    adult_mnar[missing_indices, "education_num"] <- NA
    
    cat(paste0("MNAR - education_num: ", length(missing_indices), " missing values ",
              "(biased toward lower education)\n"))
  }
  
  # MNAR Pattern 3: Certain occupations less likely to report workclass
  # (e.g., people in informal work arrangements)
  if("workclass" %in% names(adult_mnar) && "occupation" %in% names(adult_mnar)) {
    # Higher missingness for certain occupations
    high_risk_occupations <- c("Other-service", "Handlers-cleaners", "Farming-fishing", 
                              "Priv-house-serv")
    
    high_risk_indices <- which(adult_mnar$occupation %in% high_risk_occupations)
    low_risk_indices <- which(!adult_mnar$occupation %in% high_risk_occupations)
    
    n_high_risk_missing <- round(length(high_risk_indices) * 0.35)  # 35% missing
    n_low_risk_missing <- round(length(low_risk_indices) * 0.08)    # 8% missing
    
    if(n_high_risk_missing > 0) {
      missing_high_risk <- sample(high_risk_indices, n_high_risk_missing)
      adult_mnar[missing_high_risk, "workclass"] <- NA
    }
    
    if(n_low_risk_missing > 0) {
      missing_low_risk <- sample(low_risk_indices, n_low_risk_missing)
      adult_mnar[missing_low_risk, "workclass"] <- NA
    }
    
    cat(paste0("MNAR - workclass: ", n_high_risk_missing, " missing from high-risk occupations, ",
              n_low_risk_missing, " missing from other occupations\n"))
  }
  
  return(adult_mnar)
}

# ===============================
# Create both datasets
# ===============================

# Create MCAR dataset
adult_mcar <- create_mcar_data(adult_complete, missing_rate = 0.20)

# Create MNAR dataset  
adult_mnar <- create_mnar_data(adult_complete, missing_rate = 0.20)

# ===============================
# Summary statistics
# ===============================

datasets <- list("Complete" = adult_complete, "MCAR" = adult_mcar, "MNAR" = adult_mnar)

cat("\n=== MISSINGNESS COMPARISON ===\n")
for(dataset_name in names(datasets)) {
  dataset <- datasets[[dataset_name]]
  cat(paste0("\n", dataset_name, " Dataset:\n"))
  
  for(col in names(dataset)) {
    missing_count <- sum(is.na(dataset[[col]]))
    missing_pct <- round(missing_count / nrow(dataset) * 100, 1)
    if(missing_count > 0) {
      cat(paste0("  ", col, ": ", missing_count, " missing (", missing_pct, "%)\n"))
    }
  }
  
  total_missing <- sum(is.na(dataset))
  total_cells <- nrow(dataset) * ncol(dataset)
  overall_pct <- round(total_missing / total_cells * 100, 1)
  cat(paste0("  Overall: ", total_missing, " missing (", overall_pct, "%)\n"))
}

# ===============================
# Analyze MNAR patterns
# ===============================

cat("\n=== MNAR PATTERN ANALYSIS ===\n")

# Check income vs hours_per_week missingness
if("hours_per_week" %in% names(adult_mnar) && "income" %in% names(adult_mnar)) {
  missing_by_income <- table(adult_mnar$income, is.na(adult_mnar$hours_per_week))
  cat("Hours per week missingness by income:\n")
  print(missing_by_income)
  
  # Calculate percentages
  pct_missing_high <- round(missing_by_income[">50K", "TRUE"] / 
                           sum(missing_by_income[">50K", ]) * 100, 1)
  pct_missing_low <- round(missing_by_income["<=50K", "TRUE"] / 
                          sum(missing_by_income["<=50K", ]) * 100, 1)
  
  cat(paste0("High income missing rate: ", pct_missing_high, "%\n"))
  cat(paste0("Low income missing rate: ", pct_missing_low, "%\n"))
}

# Check education pattern
if("education_num" %in% names(adult_mnar)) {
  education_groups <- cut(adult_mnar$education_num, 
                         breaks = c(0, 9, 12, 16), 
                         labels = c("Low (≤9)", "Medium (10-12)", "High (>12)"),
                         include.lowest = TRUE)
  
  missing_by_education <- table(education_groups, is.na(adult_mnar$education_num), 
                               useNA = "ifany")
  cat("\nEducation missingness by education level:\n")
  print(missing_by_education)
}

# ===============================
# Save datasets
# ===============================

write.csv(adult_mcar, "adult_sample_mcar.csv", row.names = FALSE)
write.csv(adult_mnar, "adult_sample_mnar.csv", row.names = FALSE)

cat("\n=== FILES SAVED ===\n")
cat("MCAR dataset saved as 'adult_sample_mcar.csv'\n")
cat("MNAR dataset saved as 'adult_sample_mnar.csv'\n")
cat("Original complete dataset: 'adult_sample_processed.csv'\n")

# Create summary report
summary_report <- data.frame(
  Dataset = c("Complete", "MCAR", "MNAR"),
  Total_Missing = c(sum(is.na(adult_complete)), sum(is.na(adult_mcar)), sum(is.na(adult_mnar))),
  Missing_Percentage = c(
    round(sum(is.na(adult_complete)) / (nrow(adult_complete) * ncol(adult_complete)) * 100, 1),
    round(sum(is.na(adult_mcar)) / (nrow(adult_mcar) * ncol(adult_mcar)) * 100, 1),
    round(sum(is.na(adult_mnar)) / (nrow(adult_mnar) * ncol(adult_mnar)) * 100, 1)
  )
)

print(summary_report)
write.csv(summary_report, "missing_data_summary.csv", row.names = FALSE)
# Adult Dataset Preprocessing Script
# Load required libraries
library(readr)
library(dplyr)
library(utils)

# Read the adult dataset
# First, let's extract from zip if needed and read the data

adult_data <- read_csv("adult.data", col_names = FALSE)

# Standard column names for Adult dataset
col_names <- c("age", "workclass", "fnlwgt", "education", "education_num", 
               "marital_status", "occupation", "relationship", "race", "sex", 
               "capital_gain", "capital_loss", "hours_per_week", "native_country", 
               "income")

# Read the dataset (assuming it's adult.data)
adult_data <- read_csv("adult.data", 
                      col_names = col_names,
                      col_types = cols(.default = "c")) # Read as character first

# Display basic info about the dataset
cat("Original dataset dimensions:", dim(adult_data), "\n")
cat("Column names:\n")
print(names(adult_data))

# Clean the data - remove leading/trailing spaces and handle missing values
adult_data <- adult_data %>%
  mutate_all(~trimws(as.character(.))) %>%
  mutate_all(~ifelse(. == "?", NA, .))

# Convert appropriate columns to numeric
adult_data <- adult_data %>%
  mutate(
    age = as.numeric(age),
    education_num = as.numeric(education_num),
    hours_per_week = as.numeric(hours_per_week),
    capital_gain = as.numeric(capital_gain),
    capital_loss = as.numeric(capital_loss),
    fnlwgt = as.numeric(fnlwgt)
  )

# Select specific features for your analysis
selected_features <- c("age", "education_num", "hours_per_week", "sex", 
                      "marital_status", "workclass", "occupation", "income")

# Create subset with selected features
adult_subset <- adult_data %>%
  select(all_of(selected_features))

# Remove rows with missing values in selected features
adult_clean <- adult_subset %>%
  na.omit()

cat("Dataset after cleaning and feature selection:", dim(adult_clean), "\n")

# Set sample size
sample_size <- 7500  # Adjust this number if too large

# Sample the data
set.seed(42)  # For reproducibility
adult_sample <- adult_clean %>%
  slice_sample(n = min(sample_size, nrow(adult_clean)))

cat("Final sample size:", nrow(adult_sample), "\n")

# Display summary of the sample
cat("\nSummary of sampled data:\n")
summary(adult_sample)

# Display first few rows
cat("\nFirst 6 rows of the sample:\n")
head(adult_sample)

# Save the processed sample
write_csv(adult_sample, "adult_sample_processed.csv")
cat("\nProcessed sample saved as 'adult_sample_processed.csv'\n")

# Display unique values for categorical variables
cat("\nUnique values in categorical variables:\n")
for(col in c("sex", "marital_status", "workclass", "occupation", "income")) {
  if(col %in% names(adult_sample)) {
    cat(paste0(col, ": "), paste(unique(adult_sample[[col]]), collapse = ", "), "\n")
  }
}

### End of preprocessing script ###


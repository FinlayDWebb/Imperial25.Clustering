import pandas as pd
import numpy as np

# Adult Dataset Preprocessing Script
def preprocess_adult_dataset(file_path='adult.data', sample_size=7500):
    """Preprocess the Adult UCI dataset with feature selection and cleaning"""
    
    # Define column names
    col_names = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]
    
    # Read dataset
    try:
        adult_data = pd.read_csv(
            file_path,
            header=None,
            names=col_names,
            na_values='?',
            skipinitialspace=True,
            dtype={col: 'category' for col in col_names if col not in [
                'age', 'fnlwgt', 'education_num', 
                'capital_gain', 'capital_loss', 'hours_per_week'
            ]}
        )
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return
    
    # Display basic info
    print(f"Original dataset dimensions: {adult_data.shape}")
    print("\nColumn names:")
    print(adult_data.columns.tolist())
    
    # Clean data - trim whitespace for categorical columns
    cat_cols = adult_data.select_dtypes(['category', 'object']).columns
    adult_data[cat_cols] = adult_data[cat_cols].apply(lambda x: x.str.strip())
    
    # Convert income to binary classification
    adult_data['income'] = adult_data['income'].replace(
        {'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1}
    )
    
    # Select specific features
    selected_features = [
        "age", "education_num", "hours_per_week", "sex", 
        "marital_status", "workclass", "occupation", "income"
    ]
    
    adult_subset = adult_data[selected_features].copy()
    
    # Remove rows with missing values
    adult_clean = adult_subset.dropna()
    
    print(f"\nDataset after cleaning and feature selection: {adult_clean.shape}")
    
    # Sample the data
    if sample_size > len(adult_clean):
        print(f"Warning: Sample size reduced from {sample_size} to {len(adult_clean)}")
        sample_size = len(adult_clean)
    
    adult_sample = adult_clean.sample(n=sample_size, random_state=42)
    
    print(f"Final sample size: {len(adult_sample)}")
    
    # Display summary
    print("\nSummary of sampled data:")
    print(adult_sample.describe(include='all'))
    
    # Save processed sample
    output_file = "adult_sample_processed.csv"
    adult_sample.to_csv(output_file, index=False)
    print(f"\nProcessed sample saved as '{output_file}'")
    
    # Display unique values for categorical variables
    print("\nUnique values in categorical variables:")
    for col in ["sex", "marital_status", "workclass", "occupation"]:
        if col in adult_sample:
            unique_vals = adult_sample[col].unique()
            print(f"{col}: {', '.join(map(str, unique_vals))}")
    
    print("income:", adult_sample['income'].unique())
    
    return adult_sample

# Execute the preprocessing
if __name__ == "__main__":
    preprocess_adult_dataset()
#!/usr/bin/env python3
"""
MIDASpy Imputation Script
Complete implementation for missing data imputation using MIDAS neural networks
"""

import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import MIDASpy as md
from sklearn.preprocessing import MinMaxScaler
import os
import sys

# Suppress TensorFlow and NumPy warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Check NumPy version
if np.version.version >= '1.24.0':
    print(f"Using NumPy {np.__version__} - applying compatibility patches")
    np.bool = bool  # Patch for deprecated np.bool
    np.object = object  # Patch for deprecated np.object
    np.int = int  # Patch for deprecated np.int

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def identify_variable_types(data, max_categories=20):
    """
    Automatically identify variable types based on data characteristics
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    max_categories : int
        Maximum number of unique values to consider a variable categorical
        
    Returns:
    --------
    tuple: (categorical_vars, continuous_vars)
    """
    categorical_vars = []
    continuous_vars = []
    
    for col in data.columns:
        # Skip if column is entirely missing
        if data[col].isna().all():
            continue
            
        # Get unique values (excluding NaN)
        unique_vals = data[col].dropna().nunique()
        
        # Check data type
        if data[col].dtype == 'object' or data[col].dtype.name == 'category':
            categorical_vars.append(col)
        elif unique_vals <= max_categories:
            # Few unique values - likely categorical
            categorical_vars.append(col)
        else:
            # Many unique values - likely continuous
            continuous_vars.append(col)
    
    return categorical_vars, continuous_vars

def preprocess_data(data, categorical_vars):
    """
    Preprocess data for MIDAS imputation
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    categorical_vars : list
        List of categorical variable names
        
    Returns:
    --------
    tuple: (processed_data, cat_cols_list)
    """
    # Make a copy to avoid modifying original data
    data_copy = data.copy()
    
    # Separate categorical and non-categorical data
    if categorical_vars:
        # One-hot encode categorical variables
        data_cat, cat_cols_list = md.cat_conv(data_copy[categorical_vars])
        
        # Drop original categorical columns
        data_copy.drop(categorical_vars, axis=1, inplace=True)
        
        # Combine non-categorical and one-hot encoded data
        data_processed = pd.concat([data_copy, data_cat], axis=1)
    else:
        data_processed = data_copy
        cat_cols_list = []
    
    # Replace NaN values (required by MIDAS)
    na_loc = data_processed.isnull()
    data_processed[na_loc] = np.nan
    
    return data_processed, cat_cols_list

def train_midas_model(data, cat_cols_list, layer_structure=[256, 256], 
                     training_epochs=50, input_drop=0.75, seed=42):
    """
    Train MIDAS imputation model
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Preprocessed data
    cat_cols_list : list
        List of categorical column groups
    layer_structure : list
        Neural network layer structure
    training_epochs : int
        Number of training epochs
    input_drop : float
        Input dropout rate
    seed : int
        Random seed
        
    Returns:
    --------
    MIDASpy imputer object
    """
    # Initialize MIDAS imputer
    imputer = md.Midas(
        layer_structure=layer_structure,
        vae_layer=False,
        seed=seed,
        input_drop=input_drop
    )
    
    # Build model
    imputer.build_model(data, softmax_columns=cat_cols_list)
    
    # Train model
    imputer.train_model(training_epochs=training_epochs)
    
    return imputer

def generate_imputations(imputer, m=5):
    """
    Generate multiple imputations
    
    Parameters:
    -----------
    imputer : MIDASpy imputer object
        Trained MIDAS model
    m : int
        Number of imputations to generate
        
    Returns:
    --------
    list: List of imputed datasets
    """
    imputations = imputer.generate_samples(m=m).output_list
    return imputations

def convert_categorical_back(imputations, cat_cols_list, categorical_vars):
    """
    Convert one-hot encoded categorical variables back to original format
    
    Parameters:
    -----------
    imputations : list
        List of imputed datasets
    cat_cols_list : list
        List of categorical column groups
    categorical_vars : list
        Original categorical variable names
        
    Returns:
    --------
    list: List of imputed datasets with categorical variables restored
    """
    # Get flat list of all one-hot encoded columns
    flat_cats = [cat for variable in cat_cols_list for cat in variable]
    
    for i in range(len(imputations)):
        # Find the category with highest probability for each categorical variable
        tmp_cat = [imputations[i][x].idxmax(axis=1) for x in cat_cols_list]
        
        # Create DataFrame with original categorical variable names
        cat_df = pd.DataFrame({categorical_vars[j]: tmp_cat[j] for j in range(len(categorical_vars))})
        
        # Add categorical variables back and remove one-hot encoded columns
        imputations[i] = pd.concat([imputations[i], cat_df], axis=1).drop(flat_cats, axis=1)
    
    return imputations

def impute_dataset(input_file, output_prefix, m=5, layer_structure=[256, 256], 
                  training_epochs=50, categorical_vars=None, max_categories=20):
    """
    Complete imputation workflow for a single dataset
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_prefix : str
        Prefix for output files
    m : int
        Number of imputations to generate
    layer_structure : list
        Neural network layer structure
    training_epochs : int
        Number of training epochs
    categorical_vars : list or None
        List of categorical variables (if None, auto-detect)
    max_categories : int
        Maximum unique values for auto-detecting categorical variables
        
    Returns:
    --------
    list: List of imputed datasets
    """
    print(f"Processing {input_file}...")
    
    # Load data
    try:
        data = pd.read_csv(input_file)
        print(f"Data loaded: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Check for missing values
    missing_counts = data.isnull().sum()
    print(f"Missing values per column:\n{missing_counts[missing_counts > 0]}")
    
    if missing_counts.sum() == 0:
        print("No missing values found in the dataset!")
        return [data]
    
    # Identify variable types
    if categorical_vars is None:
        categorical_vars, continuous_vars = identify_variable_types(data, max_categories)
        print(f"Auto-detected categorical variables: {categorical_vars}")
        print(f"Auto-detected continuous variables: {continuous_vars}")
    
    # Preprocess data
    print("Preprocessing data...")
    data_processed, cat_cols_list = preprocess_data(data, categorical_vars)
    print(f"Processed data shape: {data_processed.shape}")
    
    # Train MIDAS model
    print("Training MIDAS model...")
    imputer = train_midas_model(
        data_processed, 
        cat_cols_list, 
        layer_structure=layer_structure,
        training_epochs=training_epochs
    )
    
    # Generate imputations
    print(f"Generating {m} imputations...")
    imputations = generate_imputations(imputer, m=m)
    
    # Convert categorical variables back to original format
    if categorical_vars:
        print("Converting categorical variables back to original format...")
        imputations = convert_categorical_back(imputations, cat_cols_list, categorical_vars)
    
    # Save imputed datasets
    print("Saving imputed datasets...")
    for i, imp_data in enumerate(imputations):
        output_file = f"{output_prefix}_imp_{i+1}.csv"
        imp_data.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
    
    # Verify imputation
    print("Verifying imputation...")
    for i, imp_data in enumerate(imputations):
        missing_after = imp_data.isnull().sum().sum()
        print(f"Imputation {i+1}: {missing_after} missing values remaining")
    
    print(f"Imputation completed for {input_file}")
    return imputations

def main():
    """
    Main function to run imputation on your datasets
    """
    # Check if required packages are installed
    try:
        import MIDASpy as md
        print("MIDASpy is available")
    except ImportError:
        print("MIDASpy not found. Please install it with: pip install MIDASpy")
        return
    
    # Define your datasets and their categorical variables
    datasets = [
        {
            'file': 'adult_sample_mcar.csv',
            'output_prefix': 'adult_mcar_midas',
            'categorical_vars': ['workclass', 'marital_status', 'occupation', 'sex', 'income']
        },
        {
            'file': 'adult_sample_mnar.csv',
            'output_prefix': 'adult_mnar_midas',
            'categorical_vars': ['workclass', 'marital_status', 'occupation', 'sex', 'income']
        }
    ]
    
    # Process each dataset
    for dataset in datasets:
        if os.path.exists(dataset['file']):
            imputations = impute_dataset(
                input_file=dataset['file'],
                output_prefix=dataset['output_prefix'],
                m=5,  # Number of imputations
                layer_structure=[128, 64],  # Neural network structure
                training_epochs=50,
                categorical_vars=dataset['categorical_vars']
            )
        else:
            print(f"File not found: {dataset['file']}")
    
    print("All processing completed!")

if __name__ == "__main__":
    main()

# Example usage for custom datasets:
"""
# For a custom dataset with auto-detection of variable types:
imputations = impute_dataset(
    input_file='your_data.csv',
    output_prefix='your_data_imputed',
    m=10,  # Generate 10 imputations
    training_epochs=100
)

# For a custom dataset with specified categorical variables:
imputations = impute_dataset(
    input_file='your_data.csv',
    output_prefix='your_data_imputed',
    categorical_vars=['gender', 'education', 'occupation'],
    m=5,
    training_epochs=50
)
"""

### Either going to use R, or Python, unsure yet which one.

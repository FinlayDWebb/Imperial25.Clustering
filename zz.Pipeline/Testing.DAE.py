#!/usr/bin/env python3

"""
MIDASpy Imputation and Pooling Script
Complete implementation for missing data imputation using MIDAS neural networks
with proper pooling based on MIDAS documentation
"""

# I might just tape together the code from the two seperate files instead of this.
# I don't trust this, and we've seen each code works on its own.

import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import MIDASpy as md
from sklearn.preprocessing import MinMaxScaler
import os
import sys
from collections import defaultdict

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
    tuple: (processed_data, cat_cols_list, scaler, continuous_vars)
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

    # Normalize continuous variables to [0,1]
    continuous_vars = [col for col in data.columns if col not in categorical_vars]
    if continuous_vars:
        scaler = MinMaxScaler()
        data_processed[continuous_vars] = scaler.fit_transform(data_processed[continuous_vars])
    else:
        scaler = None
    
    return data_processed, cat_cols_list, scaler, continuous_vars

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
    Convert one-hot encoded variables back to original categorical format
    using MIDAS's recommended method (max probability category)
    """
    # Create a mapping from category groups to original variable names
    var_mapping = {}
    for i, var in enumerate(categorical_vars):
        var_mapping[var] = cat_cols_list[i]
    
    for i in range(len(imputations)):
        # Create DataFrame for reconstructed categoricals
        cat_data = pd.DataFrame()
        
        for orig_var in categorical_vars:
            # Get all columns for this categorical variable
            var_cols = var_mapping[orig_var]
            
            if var_cols:
                # Find most probable category (max probability)
                cat_series = imputations[i][var_cols].idxmax(axis=1)
                # Remove prefix to get original category names
                cat_series = cat_series.str.replace(f"{orig_var}_", "")
                cat_data[orig_var] = cat_series
                
        # Remove one-hot columns and add reconstructed categoricals
        flat_cats = [col for cols in cat_cols_list for col in cols]
        imputations[i] = imputations[i].drop(flat_cats, axis=1)
        imputations[i] = pd.concat([imputations[i], cat_data], axis=1)
    
    return imputations

def reverse_scaling(imputations, scaler, continuous_vars):
    """Reverse MinMax scaling for continuous variables"""
    if scaler and continuous_vars:
        for i in range(len(imputations)):
            imputations[i][continuous_vars] = scaler.inverse_transform(
                imputations[i][continuous_vars])
    return imputations

def pool_imputations(imputations, categorical_vars):
    """
    Pool multiple imputations using MIDAS's recommended method:
    - For continuous variables: Average the values
    - For categorical variables: Use the most frequent category (mode)
    
    Parameters:
    -----------
    imputations : list of DataFrames
        List of imputed datasets
    categorical_vars : list
        List of categorical variable names
        
    Returns:
    --------
    pandas.DataFrame: Pooled dataset
    """
    if not imputations:
        return None
    
    # Create the pooled dataset
    pooled_df = imputations[0].copy()
    n_imp = len(imputations)
    
    # Get all columns
    all_columns = pooled_df.columns
    
    # Continuous variables (non-categorical)
    continuous_vars = [col for col in all_columns if col not in categorical_vars]
    
    # Pool continuous variables by averaging
    for col in continuous_vars:
        # Create a matrix of all imputations for this column
        values = np.array([imp[col].values for imp in imputations])
        # Average across imputations
        pooled_df[col] = np.mean(values, axis=0)
    
    # Pool categorical variables by mode (most frequent category)
    for col in categorical_vars:
        # Create a matrix of all imputations for this column
        categories = np.array([imp[col].values for imp in imputations])
        
        # Find the mode for each row
        mode_series = []
        for j in range(categories.shape[1]):
            unique, counts = np.unique(categories[:, j], return_counts=True)
            mode_series.append(unique[np.argmax(counts)])
        
        pooled_df[col] = mode_series
    
    return pooled_df

def combine_models(imputations, y_var, X_vars):
    """
    Combine models across imputations using MIDAS's combine function
    (Rubin's rules for parameter pooling)
    
    Parameters:
    -----------
    imputations : list of DataFrames
        List of imputed datasets
    y_var : str
        Dependent variable
    X_vars : list
        Independent variables
        
    Returns:
    --------
    pandas.DataFrame: Combined model results
    """
    return md.combine(y_var=y_var, X_vars=X_vars, df_list=imputations)

def impute_and_pool(input_file, output_prefix, m=5, layer_structure=[256, 256], 
                   training_epochs=50, categorical_vars=None, max_categories=20,
                   pool_imputations_flag=True, combine_model=None):
    """
    Complete imputation and pooling workflow
    
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
    pool_imputations_flag : bool
        Whether to pool the imputations into a single dataset
    combine_model : dict or None
        Parameters for model combination (Rubin's rules)
        
    Returns:
    --------
    tuple: (imputations, pooled_data, model_results)
    """
    print(f"Processing {input_file}...")
    
    # Load data
    try:
        data = pd.read_csv(input_file)
        print(f"Data loaded: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None
    
    # Check for missing values
    missing_counts = data.isnull().sum()
    print(f"Missing values per column:\n{missing_counts[missing_counts > 0]}")
    
    if missing_counts.sum() == 0:
        print("No missing values found in the dataset!")
        return [data], data, None
    
    # Identify variable types
    if categorical_vars is None:
        categorical_vars, continuous_vars = identify_variable_types(data, max_categories)
        print(f"Auto-detected categorical variables: {categorical_vars}")
        print(f"Auto-detected continuous variables: {continuous_vars}")
    else:
        continuous_vars = [col for col in data.columns if col not in categorical_vars]
    
    # Preprocess data
    print("Preprocessing data...")
    data_processed, cat_cols_list, scaler, continuous_vars = preprocess_data(data, categorical_vars)
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
    
    # Reverse scaling for continuous variables
    if scaler and continuous_vars:
        print("Reversing scaling for continuous variables...")
        imputations = reverse_scaling(imputations, scaler, continuous_vars)
    
    # Save individual imputed datasets
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
    
    # Pool imputations
    pooled_data = None
    if pool_imputations_flag and len(imputations) > 1:
        print("Pooling imputations...")
        pooled_data = pool_imputations(imputations, categorical_vars)
        pooled_file = f"{output_prefix}_pooled.csv"
        pooled_data.to_csv(pooled_file, index=False)
        print(f"Saved pooled dataset: {pooled_file}")
    
    # Combine models using Rubin's rules
    model_results = None
    if combine_model:
        print("Combining models using Rubin's rules...")
        try:
            model_results = combine_models(
                imputations,
                y_var=combine_model['y_var'],
                X_vars=combine_model['X_vars']
            )
            model_file = f"{output_prefix}_model_results.csv"
            model_results.to_csv(model_file, index=False)
            print(f"Saved model results: {model_file}")
        except Exception as e:
            print(f"Error combining models: {e}")
    
    print(f"Processing completed for {input_file}")
    return imputations, pooled_data, model_results

def main():
    """
    Main function to run imputation and pooling on datasets
    """
    # Check if required packages are installed
    try:
        import MIDASpy as md
        print("MIDASpy is available")
    except ImportError:
        print("MIDASpy not found. Please install it with: pip install MIDASpy")
        return
    
    # Define your datasets and parameters
    datasets = [
        {
            'input_file': 'adult_sample_mar.csv',  # MAR dataset
            'output_prefix': 'adult_mar_midas',
            'categorical_vars': ['workclass', 'marital_status', 'occupation', 'sex', 'income'],
            'm': 5,
            'layer_structure': [256, 256, 128],
            'training_epochs': 100,
            'combine_model': {
                'y_var': 'income',
                'X_vars': ['age', 'education_num', 'hours_per_week']
            }
        }
    ]
    
    # Process each dataset
    for dataset in datasets:
        if os.path.exists(dataset['input_file']):
            print(f"\n{'='*50}")
            print(f"Processing {dataset['input_file']}")
            print(f"{'='*50}")
            
            imputations, pooled_data, model_results = impute_and_pool(
                input_file=dataset['input_file'],
                output_prefix=dataset['output_prefix'],
                m=dataset['m'],
                layer_structure=dataset['layer_structure'],
                training_epochs=dataset['training_epochs'],
                categorical_vars=dataset.get('categorical_vars', None),
                combine_model=dataset.get('combine_model', None)
            )
            
            if model_results is not None:
                print("\nModel Results:")
                print(model_results)
        else:
            print(f"File not found: {dataset['input_file']}")
    
    print("\nAll processing completed!")

if __name__ == "__main__":
    main()

# Example usage for custom datasets:
"""
imputations, pooled_data, model_results = impute_and_pool(
    input_file='your_data_mar.csv',
    output_prefix='your_data_imputed',
    m=10,
    layer_structure=[256, 128],
    training_epochs=50,
    categorical_vars=['gender', 'education', 'occupation'],
    combine_model={'y_var': 'target', 'X_vars': ['feature1', 'feature2']}
)
"""
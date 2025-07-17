#!/usr/bin/env python3
"""
FIXED MIDASpy Imputation Script
With categorical reconstruction and scaling fixes
Now with general pooling that uses outputs from first part
"""

### Changes:
# 1. Fixed the convert back to categorical step, from one-hot back to original
# 2. Deeper training, more epochs, more layers
# 3. Normalised the continous entries into the model
# 4. Better error handling
# 5. Made pooling general - now tracks and uses outputs from first part
# This has resulted in an increase in accuracy from PFC 1, to PFC ~0.20. Huge difference.

import warnings
import numpy as np
import pandas as pd
import MIDASpy as md
from sklearn.preprocessing import MinMaxScaler
import os
import sys

# Suppress TensorFlow and NumPy warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)  # Add this line

# Check NumPy version
if np.version.version >= '1.24.0':
    print(f"Using NumPy {np.__version__} - applying compatibility patches")
    np.bool = bool
    np.object = object
    np.int = int

# Set random seed for reproducibility
np.random.seed(42)

# Global list to track processed datasets for pooling
processed_datasets = []

def identify_variable_types(data, max_categories=20):
    """Automatically identify variable types"""
    categorical_vars = []
    continuous_vars = []
    
    for col in data.columns:
        if data[col].isna().all():
            continue
        unique_vals = data[col].dropna().nunique()
        
        if data[col].dtype == 'object' or data[col].dtype.name == 'category':
            categorical_vars.append(col)
        elif unique_vals <= max_categories:
            categorical_vars.append(col)
        else:
            continuous_vars.append(col)
    
    return categorical_vars, continuous_vars

def preprocess_data(data, categorical_vars):
    """Preprocess data with proper scaling"""
    data_copy = data.copy()
    continuous_vars = [col for col in data.columns if col not in categorical_vars]
    
    # Normalize continuous variables
    scaler = MinMaxScaler(feature_range=(0, 1))
    if continuous_vars:
        data_copy[continuous_vars] = scaler.fit_transform(data_copy[continuous_vars])
    
    # One-hot encode categoricals
    if categorical_vars:
        data_cat, cat_cols_list = md.cat_conv(data_copy[categorical_vars])
        data_copy.drop(categorical_vars, axis=1, inplace=True)
        data_processed = pd.concat([data_copy, data_cat], axis=1)
    else:
        data_processed = data_copy
        cat_cols_list = []
    
    # Replace NaN values
    na_loc = data_processed.isnull()
    data_processed[na_loc] = np.nan
    
    return data_processed, cat_cols_list, scaler, continuous_vars

def convert_categorical_back(imputations, cat_cols_list, categorical_vars):
    """PROPERLY convert one-hot encoded variables to original categories"""
    # Create mapping from one-hot columns to original categories
    category_mapping = {}
    for var, cols in zip(categorical_vars, cat_cols_list):
        for col in cols:
            # Extract category name (e.g., "workclass_Private" -> "Private")
            if '_' in col:
                category = col.split('_', 1)[1]
            else:
                category = col  # Fallback if no underscore
            category_mapping[col] = (var, category)
    
    for i in range(len(imputations)):
        # Create DataFrame for reconstructed categoricals
        cat_data = pd.DataFrame(index=imputations[i].index)
        
        for orig_var in categorical_vars:
            # Get all columns for this categorical variable
            var_cols = [col for col in imputations[i].columns 
                       if col in category_mapping and category_mapping[col][0] == orig_var]
            
            if var_cols:
                # Find most probable category
                cat_series = imputations[i][var_cols].idxmax(axis=1)
                # Map to original category names
                cat_series = cat_series.map(lambda x: category_mapping[x][1] if x in category_mapping else np.nan)
                cat_data[orig_var] = cat_series
                
        # Remove one-hot columns
        flat_cats = [col for cols in cat_cols_list for col in cols]
        imputations[i] = imputations[i].drop(flat_cats, axis=1)
        
        # Add reconstructed categoricals
        imputations[i] = pd.concat([imputations[i], cat_data], axis=1)
    
    return imputations

def reverse_scaling(imputations, scaler, continuous_vars):
    """Reverse min-max scaling for continuous variables"""
    for i in range(len(imputations)):
        if continuous_vars:
            imputations[i][continuous_vars] = scaler.inverse_transform(
                imputations[i][continuous_vars])
    return imputations

def train_midas_model(data, cat_cols_list, layer_structure=[256, 256], 
                     training_epochs=100, input_drop=0.75, seed=42):
    """Train MIDAS imputation model with increased capacity"""
    imputer = md.Midas(
        layer_structure=layer_structure,
        vae_layer=False,
        seed=seed,
        input_drop=input_drop
    )
    
    imputer.build_model(data, softmax_columns=cat_cols_list)
    imputer.train_model(training_epochs=training_epochs, verbose=1)
    
    return imputer

def impute_dataset(input_file, output_prefix, m=5, layer_structure=[256, 256], 
                  training_epochs=100, categorical_vars=None, max_categories=20):
    """Complete imputation workflow with fixes"""
    print(f"\n{'='*50}")
    print(f"Processing {input_file}")
    print(f"{'='*50}")
    
    try:
        data = pd.read_csv(input_file)
        print(f"Data loaded: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Identify variable types
    if categorical_vars is None:
        categorical_vars, continuous_vars = identify_variable_types(data, max_categories)
        print(f"Auto-detected categoricals: {categorical_vars}")
        print(f"Auto-detected continuous: {continuous_vars}")
    else:
        continuous_vars = [col for col in data.columns if col not in categorical_vars]
    
    # Preprocess with scaling
    print("Preprocessing data with scaling...")
    data_processed, cat_cols_list, scaler, continuous_vars = preprocess_data(
        data, categorical_vars)
    
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
    imputations = imputer.generate_samples(m=m).output_list
    
    # Post-processing
    if categorical_vars:
        print("Converting categorical variables back...")
        imputations = convert_categorical_back(imputations, cat_cols_list, categorical_vars)
    
    print("Reversing scaling...")
    imputations = reverse_scaling(imputations, scaler, continuous_vars)
    
    # Save results and track for pooling
    print("Saving imputed datasets...")
    output_files = []
    for i, imp_data in enumerate(imputations):
        output_file = f"{output_prefix}_imp_{i+1}.csv"
        imp_data.to_csv(output_file, index=False)
        output_files.append(output_file)
        print(f"Saved: {output_file}")
    
    # Add to global tracker for pooling
    processed_datasets.append({
        'input_file': input_file,
        'output_prefix': output_prefix,
        'output_files': output_files,
        'num_imputations': m
    })
    
    # Verify no missing values remain
    for i, imp_data in enumerate(imputations):
        missing_after = imp_data.isnull().sum().sum()
        print(f"Imputation {i+1}: {missing_after} missing values remaining")
    
    print(f"Completed processing {input_file}")
    return imputations

def pool_datasets_by_averaging(file_pattern, output_filename, num_files=5):
    """
    Pool multiple CSV files by averaging imputed values across datasets.
    Assumes all datasets have the same order of records.
    
    Parameters:
    - file_pattern: string pattern for filenames (e.g., 'adult_mcar_midas_imp_{}.csv')
    - output_filename: name for the combined output file
    - num_files: number of files to pool (default: 5)
    """
    
    datasets = []
    
    # Read all datasets
    for i in range(1, num_files + 1):
        filename = file_pattern.format(i)
        
        if os.path.exists(filename):
            print(f"Reading {filename}...")
            df = pd.read_csv(filename)
            datasets.append(df)
            print(f"  - Shape: {df.shape}")
        else:
            print(f"Warning: {filename} not found!")
            return None
    
    if len(datasets) != num_files:
        print(f"Error: Expected {num_files} files, but only found {len(datasets)}")
        return None
    
    # Verify all datasets have the same shape
    base_shape = datasets[0].shape
    for i, df in enumerate(datasets[1:], 1):
        if df.shape != base_shape:
            print(f"Error: Dataset {i+1} has shape {df.shape}, expected {base_shape}")
            return None
    
    print(f"All datasets have matching shape: {base_shape}")
    
    # Create the averaged dataset
    averaged_df = datasets[0].copy()
    
    # Get numeric columns for averaging
    numeric_columns = averaged_df.select_dtypes(include=['float64', 'int64']).columns
    print(f"Numeric columns to average: {list(numeric_columns)}")
    
    # For numeric columns, average across all datasets
    for col in numeric_columns:
        values_stack = pd.concat([df[col] for df in datasets], axis=1)
        averaged_df[col] = values_stack.mean(axis=1)
    
    # For categorical columns, use the most frequent value (mode)
    categorical_columns = averaged_df.select_dtypes(include=['object']).columns
    print(f"Categorical columns (using mode): {list(categorical_columns)}")
    
    for col in categorical_columns:
        values_stack = pd.concat([df[col] for df in datasets], axis=1)
        # Get mode for each row
        averaged_df[col] = values_stack.mode(axis=1)[0]
    
    # Save the averaged dataset
    averaged_df.to_csv(output_filename, index=False)
    print(f"\nAveraged dataset saved as {output_filename}")
    print(f"Final shape: {averaged_df.shape}")
    
    return averaged_df

def pool_all_processed_datasets():
    """
    Pool all datasets that were processed in the first part
    """
    if not processed_datasets:
        print("No datasets were processed for pooling!")
        return
    
    print(f"\n{'='*70}")
    print("POOLING ALL PROCESSED DATASETS")
    print(f"{'='*70}")
    
    pooled_results = []
    
    for dataset_info in processed_datasets:
        print(f"\n{'='*50}")
        print(f"POOLING: {dataset_info['input_file']}")
        print(f"{'='*50}")
        
        # Create file pattern from output prefix
        file_pattern = dataset_info['output_prefix'] + '_imp_{}.csv'
        
        # Create pooled output filename
        pooled_filename = dataset_info['output_prefix'] + '_pooled.csv'
        
        # Pool the datasets
        pooled_data = pool_datasets_by_averaging(
            file_pattern=file_pattern,
            output_filename=pooled_filename,
            num_files=dataset_info['num_imputations']
        )
        
        if pooled_data is not None:
            pooled_results.append({
                'input_file': dataset_info['input_file'],
                'pooled_file': pooled_filename,
                'shape': pooled_data.shape,
                'data': pooled_data
            })
            print(f"✓ Successfully pooled {dataset_info['input_file']}")
        else:
            print(f"✗ Failed to pool {dataset_info['input_file']}")
    
    # Show summary of all pooled datasets
    if pooled_results:
        print(f"\n{'='*70}")
        print("FINAL POOLING SUMMARY")
        print(f"{'='*70}")
        
        for result in pooled_results:
            print(f"Input: {result['input_file']}")
            print(f"Pooled: {result['pooled_file']}")
            print(f"Shape: {result['shape']}")
            print(f"Sample data:")
            print(result['data'].head())
            print("-" * 50)
        
        print(f"\n✓ Successfully pooled {len(pooled_results)} datasets")
    else:
        print("\n✗ No datasets were successfully pooled")

def main():
    """Main function to run imputation and pooling"""
    
    # Clear the global tracker
    global processed_datasets
    processed_datasets = []
    
    # Define datasets to process
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
    
    # Process all datasets
    for dataset in datasets:
        if os.path.exists(dataset['file']):
            print(f"\n{'#'*60}")
            print(f"PROCESSING: {dataset['file']}")
            print(f"{'#'*60}")
            imputations = impute_dataset(
                input_file=dataset['file'],
                output_prefix=dataset['output_prefix'],
                m=5,
                layer_structure=[256, 256, 128],  # Deeper network
                training_epochs=100,              # More epochs
                categorical_vars=dataset['categorical_vars']
            )
        else:
            print(f"File not found: {dataset['file']}")
    
    print(f"\n{'#'*60}")
    print("PHASE 1 COMPLETE - Starting Pooling")
    print(f"{'#'*60}")
    
    # Pool all processed datasets
    pool_all_processed_datasets()
    
    print(f"\n{'#'*60}")
    print("ALL PROCESSING COMPLETED!")
    print(f"{'#'*60}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
GENERALIZED MIDASpy Imputation Script
Works with any dataset passed from Main.Pipe.r
Automatically detects categorical and continuous variables
"""

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
warnings.filterwarnings('ignore', category=UserWarning)

# Check NumPy version
if np.version.version >= '1.24.0':
    print(f"Using NumPy {np.__version__} - applying compatibility patches")
    np.bool = bool
    np.object = object
    np.int = int

# Set random seed for reproducibility
np.random.seed(42)

def identify_variable_types(data, max_categories=20):
    """Improved: Identify variable types more reliably"""
    categorical_vars = []
    continuous_vars = []
    
    for col in data.columns:
        if data[col].isna().all():
            print(f"Warning: Column '{col}' is entirely missing, skipping...")
            continue

        unique_vals = data[col].dropna().nunique()
        data_type = data[col].dtype

        # Only treat object/category types as categorical
        if data_type == 'object' or data_type.name == 'category':
            categorical_vars.append(col)
            print(f"  {col}: categorical (object/category type)")
        else:
            continuous_vars.append(col)
            print(f"  {col}: continuous ({unique_vals} unique values)")

    return categorical_vars, continuous_vars

# More faithful to MIDASpy documentation:
def preprocess_data(data, categorical_vars):
    """Preprocess data following MIDASpy documentation exactly"""
    data_copy = data.copy()
    continuous_vars = [col for col in data.columns if col not in categorical_vars]
    
    # One-hot encode categoricals using MIDASpy's method
    cat_cols_list = []
    if categorical_vars:
        print(f"One-hot encoding categorical variables: {categorical_vars}")
        # Convert to string first to handle any numeric categoricals
        for col in categorical_vars:
            data_copy[col] = data_copy[col].astype(str)
        
        data_cat, cat_cols_list = md.cat_conv(data_copy[categorical_vars])
        data_copy.drop(categorical_vars, axis=1, inplace=True)
        constructor_list = [data_copy]
        constructor_list.append(data_cat)
        data_processed = pd.concat(constructor_list, axis=1)
        print(f"Created {len([col for cols in cat_cols_list for col in cols])} one-hot columns")
    else:
        data_processed = data_copy
        print("No categorical variables to encode")
    
    # Replace NaN values as per documentation
    na_loc = data_processed.isnull()
    data_processed[na_loc] = np.nan
    
    # Return without scaler since we're not scaling
    return data_processed, cat_cols_list, continuous_vars


def convert_categorical_back(imputations, cat_cols_list, categorical_vars):
    """Convert one-hot encoded variables back using the documented method"""
    
    # In convert_categorical_back()
    if not categorical_vars or not cat_cols_list:
        return imputations  # Skip conversion if no categoricals

    # Add type check before conversion
    for i in range(len(imputations)):
        for col in categorical_vars:
            if imputations[i][col].dtype.kind not in 'iuf':  # Check if numeric
                # Skip non-numeric columns (shouldn't be categorical)
                continue
    
    print("Converting one-hot encoded variables back to categories (documented method)...")
    
    # Flatten all categorical columns as per documentation
    flat_cats = [cat for variable in cat_cols_list for cat in variable]
    
    for i in range(len(imputations)):
        try:
            # Use idxmax for each categorical variable as per documentation
            tmp_cat = [imputations[i][x].idxmax(axis=1) for x in cat_cols_list]
            
            # Create DataFrame with original categorical names
            cat_df = pd.DataFrame({categorical_vars[j]: tmp_cat[j] for j in range(len(categorical_vars))})
            
            # Concatenate and drop one-hot columns
            imputations[i] = pd.concat([imputations[i], cat_df], axis=1).drop(flat_cats, axis=1)
            
        except Exception as e:
            print(f"Warning: Error converting categoricals in dataset {i+1}: {e}")
            # Fallback: keep the one-hot encoded version
            continue
    
    return imputations


# def reverse_scaling(imputations, scaler, continuous_vars):
    """Reverse min-max scaling for continuous variables"""
    if not continuous_vars:
        return imputations
    
    print("Reversing scaling for continuous variables...")
    for i in range(len(imputations)):
        if continuous_vars:
            # Only reverse scale columns that were actually scaled
            cols_to_reverse = [col for col in continuous_vars if col in imputations[i].columns]
            if cols_to_reverse:
                imputations[i][cols_to_reverse] = scaler.inverse_transform(
                    imputations[i][cols_to_reverse])
    return imputations

def train_midas_model(data, cat_cols_list, layer_structure=[256, 256], 
                     training_epochs=100, input_drop=0.75, seed=42):
    """Train MIDAS imputation model with increased capacity"""
    print(f"Training MIDAS model with {len(layer_structure)} layers: {layer_structure}")
    
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
                  training_epochs=100, max_categories=20):
    """Complete imputation workflow - fully automated"""
    print(f"\n{'='*50}")
    print(f"Processing {input_file}")
    print(f"{'='*50}")
    
    try:
        data = pd.read_csv(input_file)
        print(f"Data loaded successfully: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Show missing data pattern
        missing_info = data.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        if len(missing_cols) > 0:
            print(f"Missing data detected:")
            for col, count in missing_cols.items():
                pct = (count / len(data)) * 100
                print(f"  {col}: {count} missing ({pct:.1f}%)")
        else:
            print("Warning: No missing data detected in input file!")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Automatically identify variable types
    print(f"\nIdentifying variable types (max_categories={max_categories}):")
    categorical_vars, continuous_vars = identify_variable_types(data, max_categories)
    
    if not categorical_vars and not continuous_vars:
        print("Error: No valid variables identified!")
        return None
    
    print(f"\nFinal variable classification:")
    print(f"  Categorical ({len(categorical_vars)}): {categorical_vars}")
    print(f"  Continuous ({len(continuous_vars)}): {continuous_vars}")
    
    # Preprocess with scaling
    print("\nPreprocessing data...")
    try:
        data_processed, cat_cols_list, continuous_vars = preprocess_data(data, categorical_vars)
        print(f"Preprocessed data shape: {data_processed.shape}")
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None
    
    # Train MIDAS model
    print("\nTraining MIDAS model...")
    try:
        imputer = train_midas_model(
            data_processed, 
            cat_cols_list, 
            layer_structure=layer_structure,
            training_epochs=training_epochs
        )
    except Exception as e:
        print(f"Error in model training: {e}")
        return None
    
    # Generate imputations
    print(f"\nGenerating {m} imputations...")
    try:
        imputations = imputer.generate_samples(m=m).output_list
        print(f"Generated {len(imputations)} imputation datasets")
    except Exception as e:
        print(f"Error generating imputations: {e}")
        return None
    
    # Post-processing
    try:
        if categorical_vars:
            print("\nConverting categorical variables back...")
            imputations = convert_categorical_back(imputations, cat_cols_list, categorical_vars)
        
        # No scaling reversal needed since we didn't scale
        
    except Exception as e:
        print(f"Error in post-processing: {e}")
        return None
    
    # Save results
    print("\nSaving imputed datasets...")
    output_files = []
    for i, imp_data in enumerate(imputations):
        output_file = f"{output_prefix}_imp_{i+1}.csv"
        
        # Ensure columns are in same order as original
        imp_data = imp_data.reindex(columns=data.columns)
        
        imp_data.to_csv(output_file, index=False)
        output_files.append(output_file)
        
        # Verify no missing values remain
        missing_after = imp_data.isnull().sum().sum()
        print(f"  Saved {output_file} - {missing_after} missing values remaining")
    
    print(f"\nCompleted processing {input_file}")
    print(f"Generated files: {output_files}")
    
    return imputations

def pool_datasets_by_averaging(file_pattern, output_filename, num_files=5):
    """Pool multiple CSV files by averaging/modal imputed values"""
    print(f"\nPooling {num_files} imputation files...")
    print(f"Pattern: {file_pattern}")
    
    datasets = []
    
    # Read all datasets
    for i in range(1, num_files + 1):
        filename = file_pattern.format(i)
        
        if os.path.exists(filename):
            print(f"Reading {filename}...")
            df = pd.read_csv(filename)
            datasets.append(df)
        else:
            print(f"Warning: {filename} not found!")
    
    if len(datasets) == 0:
        print("Error: No imputation files found!")
        return None
    
    if len(datasets) != num_files:
        print(f"Warning: Expected {num_files} files, but only found {len(datasets)}")
    
    # Verify all datasets have the same shape
    base_shape = datasets[0].shape
    for i, df in enumerate(datasets[1:], 1):
        if df.shape != base_shape:
            print(f"Error: Dataset {i+1} has shape {df.shape}, expected {base_shape}")
            return None
    
    print(f"All {len(datasets)} datasets have matching shape: {base_shape}")
    
    # Create the averaged dataset
    averaged_df = datasets[0].copy()
    
    # Get numeric columns for averaging
    numeric_columns = averaged_df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = averaged_df.select_dtypes(include=['object', 'category']).columns
    
    print(f"Numeric columns to average: {list(numeric_columns)}")
    print(f"Categorical columns (using mode): {list(categorical_columns)}")
    
    # For numeric columns, average across all datasets
    for col in numeric_columns:
        values_stack = pd.concat([df[col] for df in datasets], axis=1)
        averaged_df[col] = values_stack.mean(axis=1)
    
    # For categorical columns, use the most frequent value (mode)
    for col in categorical_columns:
        values_stack = pd.concat([df[col] for df in datasets], axis=1)
        # Get mode for each row
        mode_values = values_stack.mode(axis=1)
        if mode_values.shape[1] > 0:
            averaged_df[col] = mode_values.iloc[:, 0]  # Take first mode if multiple
        else:
            averaged_df[col] = datasets[0][col]  # Fallback to first dataset
    
    # Save the averaged dataset
    averaged_df.to_csv(output_filename, index=False)
    print(f"Averaged dataset saved as: {output_filename}")
    print(f"Final shape: {averaged_df.shape}")
    
    return averaged_df

def main():
    """Main function - processes command line arguments from R"""
    
    if len(sys.argv) != 3:
        print("Usage: python MIDAS.Pipe.py <input_file> <output_prefix>")
        print("Example: python MIDAS.Pipe.py data.csv midas_output")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_prefix = sys.argv[2]
    
    print(f"Starting MIDAS imputation pipeline...")
    print(f"Input file: {input_file}")
    print(f"Output prefix: {output_prefix}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        sys.exit(1)
    
    # Run imputation
    imputations = impute_dataset(
        input_file=input_file,
        output_prefix=output_prefix,
        m=5,  # Generate 5 imputations
        layer_structure=[256, 256, 128],  # Deep network
        training_epochs=100,              # More training
        max_categories=20                 # Threshold for categorical detection
    )
    
    if imputations is None:
        print("Error: Imputation failed!")
        sys.exit(1)
    
    # Pool the results
    print(f"\n{'='*50}")
    print("POOLING IMPUTATIONS")
    print(f"{'='*50}")
    
    pooled_data = pool_datasets_by_averaging(
        file_pattern=output_prefix + "_imp_{}.csv",
        output_filename=output_prefix + "_pooled.csv",
        num_files=5
    )
    
    if pooled_data is not None:
        print(f"\n✓ MIDAS pipeline completed successfully!")
        print(f"✓ Pooled result saved as: {output_prefix}_pooled.csv")
    else:
        print(f"\n✗ Pooling failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
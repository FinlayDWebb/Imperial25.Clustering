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
    """Automatically identify categorical and continuous variables from embedded types."""
    categorical_vars = []
    continuous_vars = []
    for col in data.columns:
        unique_vals = data[col].dropna().unique()
        if data[col].dtype.name in ['object', 'category']:
            categorical_vars.append(col)
            print(f"  {col}: Categorical (dtype: {data[col].dtype})")
        elif len(unique_vals) <= max_categories:
            categorical_vars.append(col)
            print(f"  {col}: Categorical (low cardinality: {len(unique_vals)})")
        else:
            continuous_vars.append(col)
            print(f"  {col}: Continuous")
    return categorical_vars, continuous_vars

# In impute_dataset function:
categorical_vars, continuous_vars = identify_variable_types(data, metadata_path)

def preprocess_data(data, categorical_vars):
    """Preprocess data following MIDASpy documentation exactly"""
    data_copy = data.copy()
    continuous_vars = [col for col in data.columns if col not in categorical_vars]
    
    # Store original categories before conversion
    original_categories = {}
    
    # One-hot encode categoricals using MIDASpy's method
    cat_cols_list = []
    if categorical_vars:
        print(f"One-hot encoding categorical variables: {categorical_vars}")
        for col in categorical_vars:
            data_copy[col] = data_copy[col].astype(str)
            original_categories[col] = [cat for cat in data_copy[col].unique() if cat != 'nan']
        data_cat, cat_cols_list = md.cat_conv(data_copy[categorical_vars])
        data_copy.drop(categorical_vars, axis=1, inplace=True)
        data_processed = pd.concat([data_copy, data_cat], axis=1)
        print(f"Created {len([col for cols in cat_cols_list for col in cols])} one-hot columns")
    else:
        data_processed = data_copy
        original_categories = {}
        print("No categorical variables to encode")
    
    # Replace NaN values as per documentation
    na_loc = data_processed.isnull()
    data_processed[na_loc] = np.nan
    
    return data_processed, cat_cols_list, continuous_vars, original_categories

def convert_categorical_back(imputations, cat_cols_list, categorical_vars, original_categories):
    """Convert one-hot encoded variables back using the documented method"""
    if not categorical_vars or not cat_cols_list:
        return imputations  # Skip conversion if no categoricals

    print("Converting one-hot encoded variables back to categories...")
    col_mapping = {}
    for var, onehot_cols in zip(categorical_vars, cat_cols_list):
        for col in onehot_cols:
            if col.startswith(var + "_"):
                category = col[len(var)+1:]
                col_mapping[col] = (var, category)
    for i in range(len(imputations)):
        try:
            cat_df = pd.DataFrame(index=imputations[i].index)
            for var in categorical_vars:
                var_cols = [col for col in imputations[i].columns if col.startswith(var + "_")]
                if not var_cols:
                    print(f"Warning: No one-hot columns found for variable '{var}'")
                    continue
                max_col_indices = imputations[i][var_cols].idxmax(axis=1)
                cat_df[var] = max_col_indices.map(lambda x: col_mapping.get(x, (None, None))[1])
                if cat_df[var].isnull().any():
                    fallback_category = original_categories.get(var, ['Unknown'])[0]
                    cat_df[var].fillna(fallback_category, inplace=True)
            all_onehot_cols = [col for cols in cat_cols_list for col in cols]
            existing_onehot_cols = [col for col in all_onehot_cols if col in imputations[i].columns]
            imputations[i] = imputations[i].drop(columns=existing_onehot_cols)
            imputations[i] = pd.concat([imputations[i], cat_df], axis=1)
            print(f"  Converted {len(categorical_vars)} categorical variables in dataset {i+1}")
        except Exception as e:
            print(f"Error converting categoricals in dataset {i+1}: {e}")
            print(f"Available columns: {list(imputations[i].columns)}")
            continue
    return imputations

def train_midas_model(data, cat_cols_list, layer_structure=[256, 256], 
                     training_epochs=100, input_drop=0.75, seed=42):
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
    print(f"\n{'='*50}")
    print(f"Processing {input_file}")
    print(f"{'='*50}")
    try:
        data = pd.read_feather(input_file)
        print(f"Data loaded successfully: {data.shape}")
        print(f"Columns: {list(data.columns)}")
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
    print(f"\nIdentifying variable types (max_categories={max_categories}):")
    categorical_vars, continuous_vars = identify_variable_types(data, max_categories)
    if not categorical_vars and not continuous_vars:
        print("Error: No valid variables identified!")
        return None
    print(f"\nFinal variable classification:")
    print(f"  Categorical ({len(categorical_vars)}): {categorical_vars}")
    print(f"  Continuous ({len(continuous_vars)}): {continuous_vars}")
    print("\nPreprocessing data...")
    try:
        data_processed, cat_cols_list, continuous_vars, original_categories = preprocess_data(data, categorical_vars)
        print(f"Preprocessed data shape: {data_processed.shape}")
        print(f"Original categories stored: {list(original_categories.keys())}")
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None
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
    print(f"\nGenerating {m} imputations...")
    try:
        imputations = imputer.generate_samples(m=m).output_list
        print(f"Generated {len(imputations)} imputation datasets")
    except Exception as e:
        print(f"Error generating imputations: {e}")
        return None
    try:
        if categorical_vars:
            print("\nConverting categorical variables back...")
            imputations = convert_categorical_back(
                imputations, 
                cat_cols_list, 
                categorical_vars,
                original_categories
            )
    except Exception as e:
        print(f"Error in post-processing: {e}")
        return None
    print("\nSaving imputed datasets...")
    output_files = []
    for i, imp_data in enumerate(imputations):
        output_file = f"{output_prefix}_imp_{i+1}.feather"
        imp_data = imp_data.reindex(columns=data.columns)
        imp_data.reset_index(drop=True, inplace=True)
        imp_data.to_feather(output_file)
        output_files.append(output_file)
        missing_after = imp_data.isnull().sum().sum()
        print(f"  Saved {output_file} - {missing_after} missing values remaining")
    print(f"\nCompleted processing {input_file}")
    print(f"Generated files: {output_files}")
    return imputations

def pool_datasets_by_averaging(file_pattern, output_filename, num_files=5):
    print(f"\nPooling {num_files} imputation files...")
    print(f"Pattern: {file_pattern}")
    datasets = []
    for i in range(1, num_files + 1):
        filename = file_pattern.format(i)
        if os.path.exists(filename):
            print(f"Reading {filename}...")
            df = pd.read_feather(filename)
            datasets.append(df)
        else:
            print(f"Warning: {filename} not found!")
    if len(datasets) == 0:
        print("Error: No imputation files found!")
        return None
    if len(datasets) != num_files:
        print(f"Warning: Expected {num_files} files, but only found {len(datasets)}")
    base_shape = datasets[0].shape
    for i, df in enumerate(datasets[1:], 1):
        if df.shape != base_shape:
            print(f"Error: Dataset {i+1} has shape {df.shape}, expected {base_shape}")
            return None
    print(f"All {len(datasets)} datasets have matching shape: {base_shape}")
    averaged_df = datasets[0].copy()
    numeric_columns = averaged_df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = averaged_df.select_dtypes(include=['object', 'category']).columns
    print(f"Numeric columns to average: {list(numeric_columns)}")
    print(f"Categorical columns (using mode): {list(categorical_columns)}")
    for col in numeric_columns:
        values_stack = pd.concat([df[col] for df in datasets], axis=1)
        averaged_df[col] = values_stack.mean(axis=1)
    for col in categorical_columns:
        values_stack = pd.concat([df[col] for df in datasets], axis=1)
        mode_values = values_stack.mode(axis=1)
        if mode_values.shape[1] > 0:
            averaged_df[col] = mode_values.iloc[:, 0]
        else:
            averaged_df[col] = datasets[0][col]
    averaged_df.reset_index(drop=True, inplace=True)
    averaged_df.to_feather(output_filename)
    print(f"Averaged dataset saved as: {output_filename}")
    print(f"Final shape: {averaged_df.shape}")
    return averaged_df

def main():
    if len(sys.argv) != 3:
        print("Usage: python MIDAS.Pipe.py <input_file> <output_prefix>")
        print("Example: python MIDAS.Pipe.py data.feather midas_output")
        sys.exit(1)
    input_file = sys.argv[1]
    output_prefix = sys.argv[2]
    print(f"Starting MIDAS imputation pipeline...")
    print(f"Input file: {input_file}")
    print(f"Output prefix: {output_prefix}")
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        sys.exit(1)
    imputations = impute_dataset(
        input_file=input_file,
        output_prefix=output_prefix,
        m=5,
        layer_structure=[256, 256, 128],
        training_epochs=100,
        max_categories=20
    )
    if imputations is None:
        print("Error: Imputation failed!")
        sys.exit(1)
    print(f"\n{'='*50}")
    print("POOLING IMPUTATIONS")
    print(f"{'='*50}")
    pooled_data = pool_datasets_by_averaging(
        file_pattern=output_prefix + "_imp_{}.feather",
        output_filename=output_prefix + "_pooled.feather",
        num_files=5
    )
    if pooled_data is not None:
        print(f"\n✓ MIDAS pipeline completed successfully!")
        print(f"✓ Pooled result saved as: {output_prefix}_pooled.feather")
    else:
        print(f"\n✗ Pooling failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
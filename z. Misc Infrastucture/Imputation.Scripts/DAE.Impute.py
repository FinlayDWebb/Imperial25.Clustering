#!/usr/bin/env python3
"""
FIXED MIDASpy Imputation Script
With categorical reconstruction and scaling fixes
"""

### Changes:
# 1. Fixed the convert back to categorical step, from one-hot back to original
# 2. Deeper training, more epochs, more layers
# 3. Normalised the continous entries into the model
# 4. Better error handling
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
    
    # Save results
    print("Saving imputed datasets...")
    for i, imp_data in enumerate(imputations):
        output_file = f"{output_prefix}_imp_{i+1}.csv"
        imp_data.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
    
    # Verify no missing values remain
    for i, imp_data in enumerate(imputations):
        missing_after = imp_data.isnull().sum().sum()
        print(f"Imputation {i+1}: {missing_after} missing values remaining")
    
    print(f"Completed processing {input_file}")
    return imputations

def main():
    """Main function to run imputation"""
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
    
    for dataset in datasets:
        if os.path.exists(dataset['file']):
            print(f"\n{'#'*30}")
            print(f"Processing {dataset['file']}")
            print(f"{'#'*30}")
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
    
    print("\nAll processing completed!")

if __name__ == "__main__":
    main()
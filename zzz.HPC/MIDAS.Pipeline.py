# midas_pipeline.py
import pandas as pd
import numpy as np
import MIDASpy as md
from sklearn.preprocessing import MinMaxScaler
import os
import sys

def midas_imputation(input_file, output_prefix, categorical_vars, m=5):
    # Load data
    data = pd.read_csv(input_file)
    
    # Preprocessing
    data_processed, cat_cols_list, scaler, cont_vars = preprocess_data(data, categorical_vars)
    
    # Train model
    imputer = md.Midas(layer_structure=[256, 256, 128], seed=42)
    imputer.build_model(data_processed, softmax_columns=cat_cols_list)
    imputer.train_model(training_epochs=100)
    
    # Generate imputations
    imputations = imputer.generate_samples(m=m).output_list
    
    # Post-process
    for i, imp in enumerate(imputations):
        imp = convert_categorical_back([imp], cat_cols_list, categorical_vars)[0]
        imp[cont_vars] = scaler.inverse_transform(imp[cont_vars])
        imp.to_csv(f"{output_prefix}_imp_{i+1}.csv", index=False)
    
    return imputations

def pool_imputations(pattern, num_imp=5):
    # Load imputations
    imps = [pd.read_csv(f"adult_{pattern}_midas_imp_{i+1}.csv") for i in range(num_imp)]
    
    # Pool by averaging numerics and mode for categoricals
    pooled = imps[0].copy()
    num_cols = pooled.select_dtypes(include=[np.number]).columns
    cat_cols = pooled.select_dtypes(exclude=[np.number]).columns
    
    # Numeric average
    pooled[num_cols] = pd.concat([df[num_cols] for df in imps], axis=1).mean(axis=1)
    
    # Categorical mode
    for col in cat_cols:
        pooled[col] = pd.concat([df[col] for df in imps], axis=1).mode(axis=1)[0]
    
    # Save pooled
    pooled.to_csv(f"adult_{pattern}_MIDAS_pooled.csv", index=False)
    return pooled

def main():
    datasets = [
        {"file": "adult_sample_mcar.csv", "prefix": "adult_mcar"},
        {"file": "adult_sample_mnar.csv", "prefix": "adult_mnar"}
    ]
    
    cat_vars = ["workclass", "marital_status", "occupation", "sex", "income"]
    
    for ds in datasets:
        # MIDAS imputation
        midas_imputation(ds["file"], ds["prefix"], cat_vars)
        
        # Pool results
        pattern = ds["prefix"].split("_")[-1]
        pool_imputations(pattern)
    
    print("MIDAS pipeline complete")

if __name__ == "__main__":
    main()
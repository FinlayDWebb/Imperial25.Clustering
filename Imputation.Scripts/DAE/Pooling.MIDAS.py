import pandas as pd
import os

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

# Pool MCAR datasets by averaging
print("=" * 50)
print("POOLING MCAR DATASETS BY AVERAGING")
print("=" * 50)

mcar_pooled = pool_datasets_by_averaging(
    file_pattern='adult_mcar_midas_imp_{}.csv',
    output_filename='adult_mcar_pooled.csv',
    num_files=5
)

print("\n" + "=" * 50)
print("POOLING MNAR DATASETS BY AVERAGING")
print("=" * 50)

# Pool MNAR datasets by averaging
mnar_pooled = pool_datasets_by_averaging(
    file_pattern='adult_mnar_midas_imp_{}.csv',
    output_filename='adult_mnar_pooled.csv',
    num_files=5
)

# Show sample of the pooled data
if mcar_pooled is not None:
    print("\n" + "=" * 50)
    print("SAMPLE OF MCAR POOLED DATA")
    print("=" * 50)
    print(mcar_pooled.head())
    
if mnar_pooled is not None:
    print("\n" + "=" * 50)
    print("SAMPLE OF MNAR POOLED DATA")
    print("=" * 50)
    print(mnar_pooled.head())

# Summary
if mcar_pooled is not None and mnar_pooled is not None:
    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"MCAR pooled shape: {mcar_pooled.shape}")
    print(f"MNAR pooled shape: {mnar_pooled.shape}")
    print("✓ Datasets pooled by averaging imputed values")
    
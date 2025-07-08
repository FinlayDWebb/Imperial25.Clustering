import pandas as pd
import os

def pool_datasets(file_pattern, output_filename, num_files=5):
    """
    Pool multiple CSV files into a single dataset.
    
    Parameters:
    - file_pattern: string pattern for filenames (e.g., 'adult_mcar_midas_imp_{}.csv')
    - output_filename: name for the combined output file
    - num_files: number of files to pool (default: 5)
    """
    
    combined_data = []
    
    for i in range(1, num_files + 1):
        filename = file_pattern.format(i)
        
        if os.path.exists(filename):
            print(f"Reading {filename}...")
            df = pd.read_csv(filename)
            
            # Add a column to track which imputation this came from
            df['imputation_id'] = i
            
            combined_data.append(df)
            print(f"  - Shape: {df.shape}")
        else:
            print(f"Warning: {filename} not found!")
    
    if combined_data:
        # Combine all datasets
        pooled_df = pd.concat(combined_data, ignore_index=True)
        
        # Save the combined dataset
        pooled_df.to_csv(output_filename, index=False)
        print(f"\nPooled dataset saved as {output_filename}")
        print(f"Final shape: {pooled_df.shape}")
        
        # Show some basic info
        print(f"\nDataset info:")
        print(f"- Total records: {len(pooled_df)}")
        print(f"- Records per imputation: {pooled_df['imputation_id'].value_counts().sort_index()}")
        print(f"- Columns: {list(pooled_df.columns)}")
        
        return pooled_df
    else:
        print("No files were successfully read!")
        return None

# Pool MCAR datasets
print("=" * 50)
print("POOLING MCAR DATASETS")
print("=" * 50)

mcar_pooled = pool_datasets(
    file_pattern='adult_mcar_midas_imp_{}.csv',
    output_filename='adult_mcar_pooled.csv',
    num_files=5
)

print("\n" + "=" * 50)
print("POOLING MNAR DATASETS")
print("=" * 50)

# Pool MNAR datasets
mnar_pooled = pool_datasets(
    file_pattern='adult_mnar_midas_imp_{}.csv',
    output_filename='adult_mnar_pooled.csv',
    num_files=5
)

# Optional: Show sample of the pooled data
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

# Optional: Basic comparison between the two pooled datasets
if mcar_pooled is not None and mnar_pooled is not None:
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    print(f"MCAR pooled shape: {mcar_pooled.shape}")
    print(f"MNAR pooled shape: {mnar_pooled.shape}")
    
    # Check if columns match
    mcar_cols = set(mcar_pooled.columns)
    mnar_cols = set(mnar_pooled.columns)
    
    if mcar_cols == mnar_cols:
        print("✓ Column names match between datasets")
    else:
        print("⚠ Column names differ:")
        print(f"  MCAR only: {mcar_cols - mnar_cols}")
        print(f"  MNAR only: {mnar_cols - mcar_cols}")
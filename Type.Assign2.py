import pandas as pd
import numpy as np

# Load the metadata CSV
metadata_df = pd.read_csv('australia.credit_data.clean.meta.csv')

# Normalize column names in metadata (replace hyphens with underscores)
metadata_df['variable'] = metadata_df['variable'].str.replace('-', '_')

# Convert metadata DataFrame into a dictionary
metadata = {}
for _, row in metadata_df.iterrows():
    variable = row['variable']
    type_ = row['type'].strip().lower()
    levels = row['levels']
    description = row['description']
    
    # Parse levels (handle NaN, strip whitespace)
    if pd.isna(levels):
        levels_parsed = None
    else:
        levels_parsed = [lvl.strip() for lvl in levels.split(',')]
    
    # Build dictionary entry
    metadata[variable] = {
        'type': type_,
        'levels': levels_parsed,
        'description': description
    }

input_file = 'australia.credit_data.clean.csv'
# Normalize column names in dataset too
df = pd.read_csv(input_file)
df.columns = df.columns.str.replace('-', '_')

# Check for non-numeric values in numeric columns
numeric_cols = [col for col, meta in metadata.items() if meta['type'] == 'numeric']
for col in numeric_cols:
    if col in df.columns:
        non_numeric = df[col].apply(lambda x: not str(x).strip().isdigit() if pd.notna(x) else False)
        if non_numeric.any():
            print(f"⚠️ Non-numeric values in {col}: {df[col][non_numeric].unique()}")

def apply_metadata_types(df, metadata):
    """Apply proper data types based on metadata"""
    df_processed = df.copy()
    
    for column, meta in metadata.items():
        if column not in df_processed.columns:
            print(f"Warning: Column '{column}' not found in dataset")
            continue
        
        if meta['type'] in ['categories', 'binary', 'ordered']:
            # Convert levels to integers if data appears numeric
            levels = meta['levels']
            if levels:
                try:
                    # Check if first non-null value is numeric
                    sample_val = df_processed[column].dropna().iloc[0]
                    if isinstance(sample_val, (int, float)) or str(sample_val).isdigit():
                        levels = [int(lvl) for lvl in levels]
                except (IndexError, ValueError):
                    pass
            
            # Apply categorical type
            ordered = (meta['type'] == 'ordered')
            df_processed[column] = pd.Categorical(
                df_processed[column], 
                categories=levels,
                ordered=ordered
            )
            
        elif meta['type'] == 'numeric':
            df_processed[column] = pd.to_numeric(
                df_processed[column], 
                errors='coerce'
            )
    
    return df_processed

# Apply the metadata types
df_processed = apply_metadata_types(df, metadata)

# Display information about the processed dataset
print("Dataset shape:", df_processed.shape)
print("\nColumn types after processing:")
print(df_processed.dtypes)

print("\nSample of processed data:")
print(df_processed.head())

print("\nCategorical columns summary:")
for col in df_processed.columns:
    if df_processed[col].dtype.name == 'category':
        is_ordered = df_processed[col].cat.ordered
        categories = list(df_processed[col].cat.categories)
        print(f"{col}: {'Ordered' if is_ordered else 'Nominal'} - Categories: {categories}")

# Validate data against metadata
print("\nData validation:")
for column, meta in metadata.items():
    if column in df_processed.columns:
        unique_vals = df_processed[column].dropna().unique()
        if meta['levels'] and meta['type'] in ['categories', 'binary', 'ordered']:
            # Convert metadata levels to same type as data for comparison
            expected_levels = meta['levels']
            if len(expected_levels) > 0:
                try:
                    if isinstance(unique_vals[0], (int, np.integer)):
                        expected_levels = [int(lvl) for lvl in expected_levels]
                except (IndexError, ValueError):
                    pass
            
            unexpected_vals = set(unique_vals) - set(expected_levels)
            if unexpected_vals:
                print(f"⚠️ Unexpected values in {column}: {unexpected_vals}")
            else:
                print(f"✓ {column}: All values match expected levels {expected_levels}")
        else:
            print(f"{column}: {len(unique_vals)} unique values")
        print(f"  Description: {meta['description']}")

# Save the processed dataset to CSV with proper handling of categorical data
output_file = 'py_australia_data.csv'

# FIXED: Save categoricals as text labels
df_processed.to_csv(output_file, index=False)
print(f"\n✅ Dataset processing complete! Saved to: {output_file}")
print("Note: Categorical structure is preserved in df_processed variable for analysis.")
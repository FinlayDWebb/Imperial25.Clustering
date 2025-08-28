import pandas as pd
import numpy as np

# Read the Feather file
df = pd.read_feather('ty_boston_data.feather')  # Replace with your filename
copy = df.copy()

# =============================================
# Enhanced Categorical Conversion
# =============================================
categorical_cols = ['CHAS', 'RAD', 'ZN', 'AGE']

# Convert to categorical with more bins
for col in categorical_cols:
    if col in copy.columns:
        # Enhanced binning for ZN (now 2 categories)
        if col == 'ZN':
            copy[col] = pd.cut(copy[col], 
                              bins=[0, 25, 100],  # Now only 3 edges
                              labels=['Low', 'High'])  # Now only 2 labels
        
        # Enhanced binning for AGE (4 quartile-based categories)
        elif col == 'AGE':
            copy[col] = pd.cut(copy[col], 
                              bins=[0, 25, 50, 75, 100],
                              labels=['Newest', 'Mid-New', 'Mid-Old', 'Oldest'])
        
        # RAD remains as discrete index
        else:
            copy[col] = copy[col].astype('category')

# =============================================
# Numerical Columns (unchanged)
# =============================================
numerical_cols = [
    'CRIM', 'INDUS', 'NOX', 'RM', 'DIS', 
    'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

for col in numerical_cols:
    if col in copy.columns:
        copy[col] = pd.to_numeric(copy[col], errors='coerce')

# =============================================
# Data Inspection
# =============================================
print("Column data types after conversion:")
print("=" * 60)
for col in copy.columns:
    print(f"'{col}': {copy[col].dtype}")

print("\nEnhanced AGE Distribution:")
print(copy['AGE'].value_counts().sort_index())

print("\nEnhanced ZN Distribution:")
print(copy['ZN'].value_counts().sort_index())

# Save cleaned dataset
copy.to_feather('ty_boston_data.feather')
print("\nData saved to 'boston_housing_cleaned.feather'")
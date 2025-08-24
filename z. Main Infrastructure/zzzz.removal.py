import pandas as pd

# ===========================
# Step 1: Load dataset
# ===========================
df = pd.read_feather('ty_stress_data.feather')  # replace with your filename

# Make a copy
data = df.copy()

# ===========================
# Step 3: Define categorical and numerical columns
# ===========================
categorical_cols = [
    'breathing_problem', 'noise_level', 'living_conditions', 
    'safety', 'basic_needs', 'academic_performance', 'study_load',
    'future_career_concerns'
]

numerical_cols = [
    'anxiety_level', 'self_esteem', 'depression', 'headache', 
    'blood_pressure', 'sleep_quality', 'stress_level'
]

# Convert types
for col in categorical_cols:
    if col in data.columns:
        data[col] = data[col].astype('category')

for col in numerical_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# ===========================
# Step 4: Selective binning of a few numeric columns
# ===========================
# Example: only bin anxiety, depression, and stress_level into 3 high-level bins
bin_columns = {
    'anxiety_level': 3,      # low/medium/high
    'depression': 3,
    'stress_level': 3
}

for col, bins in bin_columns.items():
    if col in data.columns:
        # Create binned categorical column
        data[col + '_cat'] = pd.qcut(data[col], q=bins, labels=False, duplicates='drop')
        data[col + '_cat'] = data[col + '_cat'].astype('category')

# ===========================
# Step 5: Display info
# ===========================
print("Column data types after conversion and binning:")
print("="*60)
for col in data.columns:
    print(f"'{col}': {data[col].dtype}")

print(f"\nDataset shape: {data.shape}")
print(f"Total columns: {len(data.columns)}")

# Check for missing values
print("\nMissing values per column:")
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])

# Display categorical distributions
print("\nCategorical distributions:")
for col in data.select_dtypes(include='category').columns:
    print(f"\n{col}:")
    print(data[col].value_counts())

# Display basic statistics for remaining numerical variables
remaining_numerical = data.select_dtypes(include=['float64', 'int64']).columns
if len(remaining_numerical) > 0:
    print("\nBasic statistics for numerical variables:")
    print(data[remaining_numerical].describe())

# ===========================
# Step 6: Save cleaned dataset
# ===========================
data.to_feather('mental_health_cleaned.feather')
print("\nData saved to 'mental_health_cleaned.feather'")

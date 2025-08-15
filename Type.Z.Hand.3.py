import pandas as pd

# Load the dataset
df = pd.read_csv('StressLevelDataset.csv')

# Make a copy
copy = df.copy()

# Categorical columns (if any categorical columns exist, add them here)
categorical_cols = [
    # You can include columns like 'mental_health_history', 'bullying', etc. if they are categorical
    'mental_health_history', 
    'teacher_student_relationship', 
    'social_support',
    'peer_pressure',
    'extracurricular_activities',
    'bullying'
]

for col in categorical_cols:
    if col in copy.columns:
        copy[col] = copy[col].astype('category')

# Numerical columns
numerical_cols = [
    'anxiety_level',
    'self_esteem',
    'depression',
    'headache',
    'blood_pressure',
    'sleep_quality',
    'breathing_problem',
    'noise_level',
    'living_conditions',
    'safety',
    'basic_needs',
    'academic_performance',
    'study_load',
    'future_career_concerns',
    'stress_level'
]

for col in numerical_cols:
    if col in copy.columns:
        copy[col] = pd.to_numeric(copy[col], errors='coerce')

# Display column types after conversion
print("Column data types after conversion:")
print("=" * 60)
for col in copy.columns:
    print(f"'{col}': {copy[col].dtype}")

print(f"\nDataset shape: {copy.shape}")
print(f"Total columns: {len(copy.columns)}")

# Display sample data
print("\nFirst 5 rows:")
print(copy.head())

# Check for missing values
print("\nMissing values per column:")
missing_values = copy.isnull().sum()
print(missing_values[missing_values > 0])

# Display categorical value counts
print("\nCategorical distributions:")
for col in categorical_cols:
    if col in copy.columns:
        print(f"\n{col}:")
        print(copy[col].value_counts())

# Display basic statistics for numerical variables
print("\nBasic statistics for numerical variables:")
available_numerical = [col for col in numerical_cols if col in copy.columns]
if available_numerical:
    print(copy[available_numerical].describe())

# Save cleaned dataset to feather format for faster future loading
copy.to_feather('ty_stress_data.feather')

print("\nData saved as:")
print(" - 'mental_health_data_cleaned.feather'")
print(" - 'mental_health_data_cleaned.csv'")

import pandas as pd

# Read the CSV file (adjust filename as needed)
df = pd.read_csv('weather.class_data.clean.csv')

# Make a copy
copy = df.copy()

# Categorical columns
categorical_cols = [
    'Cloud Cover',
    'Season',
    'Location',
    'Weather Type'  # target variable
]

for col in categorical_cols:
    if col in copy.columns:
        copy[col] = copy[col].astype('category')

# Numerical columns
numerical_cols = [
    'Temperature',
    'Humidity',
    'Wind Speed',
    'Precipitation (%)',
    'Atmospheric Pressure',
    'UV Index',
    'Visibility (km)'
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

# Save cleaned dataset to feather format
copy.to_feather('ty_weather_data.feather')
print("\nData saved to 'weather_data_processed.feather'")

import pandas as pd
import numpy as np

# Read the feather file
df = pd.read_feather('ty_stars_data.feather')

# Read the CSV file
# df = pd.read_csv('Manila.Flood.csv') 


# Define additional missing value representations
missing_values = ["?", "NA", "N/A", "na", "n/a", "", "null", "NULL", "NaN", "nan", "NAN", "None", "none"]

# Replace these values with actual NaN
df = df.replace(missing_values, np.nan)

# Drop rows with any missing values
df_clean = df.dropna()

print(f"Original shape: {df.shape}")
print(f"After removing missing rows: {df_clean.shape}")

# Save to a new feather file
df_clean.to_feather('ty_stars_data.feather')
# print("Cleaned data saved to 'cleaned_data.feather'")

# Save to a new CSV file
# df_clean.to_csv('cleaned.adult.income.csv', index=False)
# print("Cleaned data saved to 'cleaned.adult.income.csv'")

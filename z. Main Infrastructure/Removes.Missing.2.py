import pandas as pd

# Read the feather file
df = pd.read_feather('ty_stress_data.feather')

# Drop the column if it exists
col_to_delete = 'mental_health_history'

if col_to_delete in df.columns:
    df = df.drop(columns=[col_to_delete])
    print(f"Column '{col_to_delete}' deleted.")
else:
    print(f"Column '{col_to_delete}' not found in dataset.")

# Save to a new feather file
df.to_feather('ty_stress_data.feather')
print("Data saved to 'vitamin_c_removed.feather'")

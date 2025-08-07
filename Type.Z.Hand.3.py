import pandas as pd

df = pd.read_csv('py_adult_data.csv')

# Make a copy
copy = df.copy()

# Categorical columns (these will behave like strings/categories)
copy['marital_status'] = copy['marital_status'].astype('category')
copy['workclass'] = copy['workclass'].astype('category')
copy['occupation'] = copy['occupation'].astype('category')

# Binary column (Male/Female)
copy['sex'] = copy['sex'].astype('category')

# Binary target column (0/1)
copy['income_greater_than_50k'] = copy['income_greater_than_50k'].astype('category')

# Numeric columns (float or int)
copy['age'] = copy['age'].astype(float)
copy['education_num'] = copy['education_num'].astype(float)
copy['hours_per_week'] = copy['hours_per_week'].astype(float)



for col in copy.columns:
    print(f"{col}: {copy[col].dtype}")

# Save to new CSV
copy.to_feather('ty_adult_data.feather')

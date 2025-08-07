import pandas as pd

df = pd.read_csv('wine.red_data.clean.csv')

# Make a copy
copy = df.copy()

# Numeric columns
copy['fixed_acidity'] = copy['fixed_acidity'].astype(float)
copy['volatile_acidity'] = copy['volatile_acidity'].astype(float)
copy['citric_acid'] = copy['citric_acid'].astype(float)
copy['residual_sugar'] = copy['residual_sugar'].astype(float)
copy['chlorides'] = copy['chlorides'].astype(float)
copy['free_sulfur_dioxide'] = copy['free_sulfur_dioxide'].astype(float)
copy['total_sulfur_dioxide'] = copy['total_sulfur_dioxide'].astype(float)
copy['density'] = copy['density'].astype(float)
copy['pH'] = copy['pH'].astype(float)
copy['sulphates'] = copy['sulphates'].astype(float)
copy['alcohol'] = copy['alcohol'].astype(float)

# Categorical (ordered target)
copy['quality'] = copy['quality'].astype('category')


for col in copy.columns:
    print(f"{col}: {copy[col].dtype}")

# Save to new CSV
copy.to_csv('py_wine.red_data.csv', index=False)

import pandas as pd

df = pd.read_csv('wisconsin.UCI__data.clean.csv')

# Make a copy
copy = df.copy()

# Numeric columns (mean features)
copy['radius_mean'] = copy['radius_mean'].astype(float)
copy['texture_mean'] = copy['texture_mean'].astype(float)
copy['perimeter_mean'] = copy['perimeter_mean'].astype(float)
copy['area_mean'] = copy['area_mean'].astype(float)
copy['smoothness_mean'] = copy['smoothness_mean'].astype(float)
copy['compactness_mean'] = copy['compactness_mean'].astype(float)
copy['concavity_mean'] = copy['concavity_mean'].astype(float)
copy['concave points_mean'] = copy['concave points_mean'].astype(float)
copy['symmetry_mean'] = copy['symmetry_mean'].astype(float)
copy['fractal_dimension_mean'] = copy['fractal_dimension_mean'].astype(float)

# Numeric columns (standard error features)
copy['radius_se'] = copy['radius_se'].astype(float)
copy['texture_se'] = copy['texture_se'].astype(float)
copy['perimeter_se'] = copy['perimeter_se'].astype(float)
copy['area_se'] = copy['area_se'].astype(float)
copy['smoothness_se'] = copy['smoothness_se'].astype(float)
copy['compactness_se'] = copy['compactness_se'].astype(float)
copy['concavity_se'] = copy['concavity_se'].astype(float)
copy['concave points_se'] = copy['concave points_se'].astype(float)
copy['symmetry_se'] = copy['symmetry_se'].astype(float)
copy['fractal_dimension_se'] = copy['fractal_dimension_se'].astype(float)

# Numeric columns (worst features)
copy['radius_worst'] = copy['radius_worst'].astype(float)
copy['texture_worst'] = copy['texture_worst'].astype(float)
copy['perimeter_worst'] = copy['perimeter_worst'].astype(float)
copy['area_worst'] = copy['area_worst'].astype(float)
copy['smoothness_worst'] = copy['smoothness_worst'].astype(float)
copy['compactness_worst'] = copy['compactness_worst'].astype(float)
copy['concavity_worst'] = copy['concavity_worst'].astype(float)
copy['concave points_worst'] = copy['concave points_worst'].astype(float)
copy['symmetry_worst'] = copy['symmetry_worst'].astype(float)
copy['fractal_dimension_worst'] = copy['fractal_dimension_worst'].astype(float)

# Binary target
copy['diagnosis'] = copy['diagnosis'].astype('category')  # B=benign, M=malignant


for col in copy.columns:
    print(f"{col}: {copy[col].dtype}")

# Save to new CSV
copy.to_csv('py_wisconsin_data.csv', index=False)

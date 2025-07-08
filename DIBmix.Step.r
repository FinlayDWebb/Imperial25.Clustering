### Here the outline is to do the following:
# 1. Load the data
# 2. Cluster the original dataset
# 3. Cluster the imputed datasets
# 4. Compare the ARI for each

# This works, because instead of measuring the imputed data against the datasets ground truths (i.e. categories)
# We instead test it relatively against the clustererd data before imputation, that becomes our ground truth.
# This means that how well imputation preserves the natural structure of the data, rather than, how well the imputation
# preserves the supervised learning potential of the data. 

# We want to know strenghts/weaknesses with general datatypes, non-linearity, homogeneity, etc. Not how well imputation
# preserves the ability to predict a target variable (like income, or age, etc.).

# Load necessary libraries

# Do this with DIBmix, and note we have to fix the bandwidths in DIBmix. But let us first find a good bandwidth that works.


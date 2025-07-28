import csv

# Define the column headers
headers = [
    'CRIM',    # Per capita crime rate by town
    'ZN',      # Proportion of residential land zoned for lots over 25,000 sq.ft.
    'INDUS',   # Proportion of non-retail business acres per town
    'CHAS',    # Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    'NOX',     # Nitric oxides concentration (parts per 10 million)
    'RM',      # Average number of rooms per dwelling
    'AGE',     # Proportion of owner-occupied units built prior to 1940
    'DIS',     # Weighted distances to five Boston employment centers
    'RAD',     # Index of accessibility to radial highways
    'TAX',     # Full-value property-tax rate per $10,000
    'PTRATIO', # Pupil-teacher ratio by town
    'B',       # 1000(Bk−0.63)² where Bk is proportion of blacks by town
    'LSTAT',   # % lower status of the population
    'MEDV'     # Median value of owner-occupied homes in $1000s
]

input_file = 'boston.housing_data.clean.csv'
output_file = 'boston_housing_with_headers.csv'

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    # Create a CSV writer
    writer = csv.writer(outfile)
    
    # Write the header row
    writer.writerow(headers)
    
    # Process each line of the input file
    for line in infile:
        # Split the line by whitespace and remove any empty strings
        values = [x for x in line.strip().split(' ') if x]
        
        # Verify we have the correct number of columns
        if len(values) == len(headers):
            writer.writerow(values)
        else:
            print(f"Skipping malformed row: {values}")

print(f"Successfully created {output_file} with proper headers.")
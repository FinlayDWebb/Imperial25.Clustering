import csv

input_file = 'retail_sales_dataset.csv'
output_file = 'retail_sales_dataset_cleaned.csv'

with open(input_file, 'r', newline='') as infile, \
     open(output_file, 'w', newline='') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    headers = next(reader)  # Read header row
    # Keep only columns starting from Gender (index 3 onward)
    new_headers = headers[3:]
    writer.writerow(new_headers)
    
    for row in reader:
        # Skip first three values (ID, Date, Customer ID)
        cleaned_row = row[3:]
        writer.writerow(cleaned_row)

print("Cleaned file saved as:", output_file)
print("Removed columns: Transaction ID, Date, Customer ID")
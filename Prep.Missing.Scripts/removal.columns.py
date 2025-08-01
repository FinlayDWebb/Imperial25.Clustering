import csv

input_file = 'w.csv'
output_file = 'w.1.csv'

with open(input_file, 'r', newline='') as infile, \
     open(output_file, 'w', newline='') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    headers = next(reader)  # Read header row
    # Keep only columns starting from index 1 onwards
    new_headers = headers[1:]
    writer.writerow(new_headers)
    
    for row in reader:
        # Skip first value, change to index accordingly
        cleaned_row = row[1:]
        writer.writerow(cleaned_row)

print("Cleaned file saved as:", output_file)
print("Removed columns: Transaction ID, Date, Customer ID")
import csv
import sys

def clean_csv(input_file, output_file):
    """Remove rows with missing values ('?') from a CSV file"""
    cleaned_rows = []
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Save header
        
        for row in reader:
            # Skip rows with any missing values
            if '?' not in row:
                cleaned_rows.append(row)
    
    # Write cleaned data to new file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(cleaned_rows)
    
    return len(cleaned_rows)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_csv.py <input_file.csv> <output_file.csv>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    num_clean_rows = clean_csv(input_path, output_path)
    print(f"Cleaned data saved to {output_path}")
    print(f"Original rows: {num_clean_rows + sum(1 for _ in open(input_path)) - num_clean_rows - 1}")
    print(f"Clean rows: {num_clean_rows}")
    print(f"Rows removed: {sum(1 for _ in open(input_path)) - num_clean_rows - 1}")
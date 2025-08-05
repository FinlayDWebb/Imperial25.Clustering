import csv
import sys

def clean_csv(input_file, output_file):
    """Remove rows with missing (empty) values from a CSV file."""
    cleaned_rows = []
    total_rows = 0
    removed_rows = 0

    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        cleaned_rows.append(header)

        for row in reader:
            total_rows += 1
            if '' in row:
                removed_rows += 1
                continue
            cleaned_rows.append(row)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(cleaned_rows)

    return total_rows, total_rows - removed_rows, removed_rows

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_csv.py <input_file.csv> <output_file.csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    total, clean, removed = clean_csv(input_path, output_path)
    print(f"Cleaned data saved to {output_path}")
    print(f"Original rows: {total}")
    print(f"Clean rows: {clean}")
    print(f"Rows removed: {removed}")


import csv

headers = [
    "temperature",
    "nausea",
    "lumbar_pain",
    "urine_pushing",
    "micturition_pains",
    "urethra_burning",
    "inflammation",
    "nephritis"
]

# Read with correct encoding (latin-1) and write CSV
with open('diagnosis.data', 'r', encoding='latin-1') as infile, \
     open('diagnosis.csv', 'w', newline='') as outfile:

    writer = csv.writer(outfile)
    writer.writerow(headers)

    for line in infile:
        line = line.strip()
        if not line:
            continue
        # Handle decimal comma and tab separation
        parts = line.replace(',', '.', 1).split('\t')
        writer.writerow(parts)

print("CSV file created: diagnosis.csv")
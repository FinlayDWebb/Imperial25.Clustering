import pandas as pd
import re
import os

def arff_to_csv(arff_file_path, csv_file_path=None):
    """
    Convert ARFF file to CSV format.
    
    Parameters:
    arff_file_path (str): Path to the input ARFF file
    csv_file_path (str): Path for the output CSV file (optional)
    
    Returns:
    pd.DataFrame: The converted data as a pandas DataFrame
    """
    
    # If no output path specified, create one based on input file
    if csv_file_path is None:
        base_name = os.path.splitext(arff_file_path)[0]
        csv_file_path = base_name + '.csv'
    
    with open(arff_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Remove whitespace and empty lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Parse ARFF file
    attributes = []
    data_section = False
    data_lines = []
    
    for line in lines:
        # Skip comments
        if line.startswith('%'):
            continue
            
        # Convert to lowercase for case-insensitive matching
        line_lower = line.lower()
        
        # Check for relation (dataset name)
        if line_lower.startswith('@relation'):
            relation_name = line.split()[1] if len(line.split()) > 1 else "dataset"
            print(f"Converting dataset: {relation_name}")
            continue
            
        # Parse attributes
        if line_lower.startswith('@attribute'):
            # Extract attribute name (handle quoted names)
            if '"' in line:
                # Handle quoted attribute names
                attr_match = re.search(r'@attribute\s+"([^"]+)"', line, re.IGNORECASE)
                if attr_match:
                    attr_name = attr_match.group(1)
                else:
                    # Fallback parsing
                    parts = line.split()
                    attr_name = parts[1].strip('"')
            else:
                # Simple attribute name without quotes
                parts = line.split()
                attr_name = parts[1]
            
            attributes.append(attr_name)
            continue
            
        # Check for data section
        if line_lower.startswith('@data'):
            data_section = True
            continue
            
        # Collect data lines
        if data_section:
            # Skip empty lines in data section
            if line:
                data_lines.append(line)
    
    print(f"Found {len(attributes)} attributes: {attributes}")
    print(f"Found {len(data_lines)} data rows")
    
    # Parse data lines
    data_rows = []
    for line in data_lines:
        # Handle different data formats
        if line.startswith('{') and line.endswith('}'):
            # Sparse format: {index value, index value, ...}
            sparse_data = ['?'] * len(attributes)  # Initialize with missing values
            content = line[1:-1]  # Remove braces
            if content.strip():  # Only process non-empty content
                pairs = content.split(',')
                for pair in pairs:
                    pair = pair.strip()
                    if ' ' in pair:
                        index_str, value = pair.split(' ', 1)
                        try:
                            index = int(index_str)
                            if 0 <= index < len(attributes):
                                sparse_data[index] = value.strip()
                        except ValueError:
                            continue
            data_rows.append(sparse_data)
        else:
            # Dense format: comma-separated values
            # Handle quoted values and commas within quotes
            values = []
            current_value = ""
            in_quotes = False
            
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                    current_value += char
                elif char == ',' and not in_quotes:
                    values.append(current_value.strip().strip('"'))
                    current_value = ""
                else:
                    current_value += char
            
            # Add the last value
            if current_value:
                values.append(current_value.strip().strip('"'))
            
            # Ensure we have the right number of columns
            while len(values) < len(attributes):
                values.append('?')  # Missing value marker
            
            data_rows.append(values[:len(attributes)])  # Trim extra columns
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=attributes)
    
    # Replace ARFF missing value markers with pandas NaN
    df = df.replace(['?', ''], pd.NA)
    
    # Try to infer data types
    for col in df.columns:
        # Try to convert to numeric first
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except:
            pass
    
    # Save to CSV
    df.to_csv(csv_file_path, index=False)
    print(f"Successfully converted ARFF to CSV: {csv_file_path}")
    print(f"DataFrame shape: {df.shape}")
    
    # Display sample data
    print("\nFirst 5 rows of converted data:")
    print(df.head())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    return df

# Usage examples:
if __name__ == "__main__":
    # Example 1: Convert single file (specify both input and output)
    df = arff_to_csv('Autism-Child-Data.arff', 'autism.child.csv')
    
    # Example 2: Convert single file (auto-generate output filename)
    # df = arff_to_csv('input_file.arff')
    
    # Example 3: Batch convert multiple ARFF files
    def batch_convert_arff_to_csv(input_folder, output_folder=None):
        """
        Convert all ARFF files in a folder to CSV format.
        """
        if output_folder is None:
            output_folder = input_folder
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        converted_files = []
        for filename in os.listdir(input_folder):
            if filename.lower().endswith('.arff'):
                input_path = os.path.join(input_folder, filename)
                output_filename = os.path.splitext(filename)[0] + '.csv'
                output_path = os.path.join(output_folder, output_filename)
                
                try:
                    print(f"\n{'='*50}")
                    print(f"Converting: {filename}")
                    arff_to_csv(input_path, output_path)
                    converted_files.append(output_filename)
                except Exception as e:
                    print(f"Error converting {filename}: {str(e)}")
        
        print(f"\n{'='*50}")
        print(f"Batch conversion complete!")
        print(f"Converted {len(converted_files)} files:")
        for file in converted_files:
            print(f"  - {file}")
    
    # Uncomment to use batch conversion:
    # batch_convert_arff_to_csv('path/to/arff/files', 'path/to/output/csvs')
    
    # For single file conversion, update the filename below:
    print("To convert a file, call:")
    print("df = arff_to_csv('your_file.arff')")
    print("\nOr for batch conversion:")
    print("batch_convert_arff_to_csv('folder_with_arff_files')")
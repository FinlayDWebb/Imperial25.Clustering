import pandas as pd
import os
from pathlib import Path

def sample_datasets(input_folder='Processed.Data.Temp', max_rows=1000):
    """
    Process feather files in the specified folder, sampling them to max_rows if they're larger,
    and save them with a 'sample.' prefix.
    
    Args:
        input_folder (str): Path to the folder containing feather files
        max_rows (int): Maximum number of rows to keep (default: 1000)
    """
    
    # Create Path object for the input folder
    folder_path = Path(input_folder)
    
    # Check if folder exists
    if not folder_path.exists():
        print(f"Error: Folder '{input_folder}' does not exist.")
        return
    
    # Find all feather files in the folder
    feather_files = list(folder_path.glob("*.feather"))
    
    if not feather_files:
        print(f"No feather files found in '{input_folder}'")
        return
    
    print(f"Found {len(feather_files)} feather files to process...")
    
    for file_path in feather_files:
        try:
            # Read the feather file
            print(f"Processing: {file_path.name}")
            df = pd.read_feather(file_path)
            
            original_rows = len(df)
            print(f"  Original rows: {original_rows}")
            
            # Sample if necessary
            if original_rows > max_rows:
                # Random sample of max_rows
                df_sampled = df.sample(n=max_rows, random_state=42)
                print(f"  Sampled to: {max_rows} rows")
            else:
                df_sampled = df
                print(f"  Kept all rows (≤ {max_rows})")
            
            # Create output filename with 'sample.' prefix
            output_filename = f"sample.{file_path.name}"
            output_path = folder_path / output_filename
            
            # Save the sampled dataset
            df_sampled.to_feather(output_path)
            print(f"  Saved as: {output_filename}")
            print()
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
            print()
    
    print("Processing complete!")

if __name__ == "__main__":
    # Run the sampling function
    sample_datasets()
    
    # Optional: You can also call it with different parameters
    # sample_datasets('your_folder_name', 500)  # Different folder and max rows
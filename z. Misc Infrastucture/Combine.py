import pandas as pd
import os
import glob
import sys

def combine_feathers(input_folder="zzzz.Combine", output_file="combined.csv"):
    # Find all feather files in the folder
    feather_files = glob.glob(os.path.join(input_folder, "*.feather"))
    
    if not feather_files:
        print("No feather files found in", input_folder)
        return
    
    # Read and append
    dfs = []
    for file in sorted(feather_files):  # sorted ensures consistent order
        print(f"Reading {file}...")
        df = pd.read_feather(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    print(f"✅ Combined file saved as {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Combine.py output_filename.csv")
    else:
        output_filename = sys.argv[1]
        combine_feathers(output_file=output_filename)

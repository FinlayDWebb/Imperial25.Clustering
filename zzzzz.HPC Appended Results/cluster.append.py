import pandas as pd
import os
import glob
import re

def load_imputation_winners(imputation_results_file):
    """
    Load the imputation winners from the previous results.
    
    Args:
        imputation_results_file (str): Path to imputation results CSV
    
    Returns:
        pandas DataFrame: DataFrame with imputation winners
    """
    if not os.path.exists(imputation_results_file):
        print(f"Warning: Imputation results file '{imputation_results_file}' not found.")
        return pd.DataFrame()
    
    df = pd.read_csv(imputation_results_file)
    # Filter only for specific missing rate categories, not overall
    df = df[df['Category'].str.startswith('MissingRate_')]
    # Extract missing rate from category
    df['MissingRate'] = df['Category'].str.replace('MissingRate_', '').astype(float)
    return df[['Dataset', 'MissingRate', 'Winner_Method']]

def extract_method_from_filename(filename):
    """
    Extract the method name from the filename in the File column.
    Example: 'ty_hepatitis_data_famd_0.05_imputed.feather' -> 'FAMD'
    """
    # Look for method names in the filename
    filename_lower = filename.lower()
    if 'mice' in filename_lower:
        return 'MICE'
    elif 'famd' in filename_lower:
        return 'FAMD'
    elif 'missforest' in filename_lower or 'miss_forest' in filename_lower:
        return 'missForest'
    return None

def calculate_cluster_winners(df, dataset_name, n_clusters, imputation_winners):
    """
    Calculate the best performing method for clustering results.
    
    Args:
        df (pandas DataFrame): Input clustering data
        dataset_name (str): Name of the dataset
        n_clusters (float): Number of clusters
        imputation_winners (DataFrame): Imputation winners data
    
    Returns:
        list: List of dictionaries with winner information
    """
    results = []
    
    # First, extract the actual method names from the File column
    df['Actual_Method'] = df['File'].apply(extract_method_from_filename)
    
    # Get unique missing rates
    missing_rates = df['MissingRate'].unique()
    
    # Process each missing rate category
    for missing_rate in missing_rates:
        # Filter data for this missing rate
        df_subset = df[df['MissingRate'] == missing_rate]
        
        # Find the method with highest ARI for this missing rate
        winner_idx = df_subset['ARI'].idxmax()
        winner = df_subset.loc[winner_idx]
        
        # Find the precluster winner's ARI (imputation stage winner)
        precluster_winner_method = None
        precluster_winner_ari = None
        
        if not imputation_winners.empty:
            imputation_winner = imputation_winners[
                (imputation_winners['Dataset'] == dataset_name) & 
                (imputation_winners['MissingRate'] == missing_rate)
            ]
            
            if not imputation_winner.empty:
                precluster_winner_method = imputation_winner['Winner_Method'].iloc[0]
                # Find the ARI for the precluster winner method
                precluster_data = df_subset[df_subset['Actual_Method'] == precluster_winner_method]
                if not precluster_data.empty:
                    precluster_winner_ari = precluster_data['ARI'].iloc[0]
        
        results.append({
            'Dataset': dataset_name,
            'NClusters': n_clusters,
            'MissingRate': missing_rate,
            'Winner_Method': winner['Actual_Method'],
            'Winner_ARI': winner['ARI'],
            'Precluster_Winner_Method': precluster_winner_method,
            'Precluster_Winner_ARI': precluster_winner_ari,
            'Category': f'MissingRate_{missing_rate}'
        })
    
    return results

def parse_filename(filename):
    """
    Parse dataset name and number of clusters from filename.
    Handles filenames like: Dataset.number.csv, DatasetWithDots.2.csv, Wine.White.3.csv
    """
    # Remove .csv extension
    basename = filename.replace('.csv', '')
    parts = basename.split('.')
    
    # Try different parsing strategies
    if len(parts) >= 2:
        # Try to convert the last part to float (cluster number)
        try:
            n_clusters = float(parts[-1])
            # The rest is the dataset name
            dataset_name = '.'.join(parts[:-1])
            return dataset_name, n_clusters
        except ValueError:
            pass
    
    # Alternative parsing for files like "Wine.White.2.csv"
    # Look for a numeric part at the end
    for i in range(len(parts)):
        try:
            n_clusters = float(parts[i])
            dataset_name = '.'.join(parts[:i])
            return dataset_name, n_clusters
        except ValueError:
            continue
    
    return None, None

def process_cluster_datasets(folder_path, imputation_winners_file):
    """
    Process all cluster datasets in the specified folder.
    """
    results = []
    
    # Load imputation winners
    imputation_winners = load_imputation_winners(imputation_winners_file)
    
    # Find all CSV files in the folder
    pattern = os.path.join(folder_path, '*.csv')
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return results
    
    print(f"Found {len(csv_files)} cluster datasets to process...")
    
    for file_path in csv_files:
        try:
            # Extract dataset name and number of clusters from filename
            filename = os.path.basename(file_path)
            dataset_name, n_clusters = parse_filename(filename)
            
            if dataset_name is None or n_clusters is None:
                print(f"Skipping {filename}: could not parse dataset name and cluster number")
                continue
            
            print(f"Processing: {dataset_name} (NClusters: {n_clusters})")
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if the required columns exist
            required_columns = ['File', 'MissingRate', 'ARI']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"  Skipping: Missing columns {missing_columns}")
                continue
            
            # Calculate cluster winners
            dataset_results = calculate_cluster_winners(df, dataset_name, n_clusters, imputation_winners)
            results.extend(dataset_results)
            
            # Print summary for this dataset
            for result in dataset_results:
                print(f"  MissingRate {result['MissingRate']}: {result['Winner_Method']} (ARI: {result['Winner_ARI']:.4f})")
                if result['Precluster_Winner_ARI'] is not None:
                    print(f"    Precluster winner ({result['Precluster_Winner_Method']}): ARI {result['Precluster_Winner_ARI']:.4f}")
                else:
                    print(f"    Precluster winner: Not found in data")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    return results

def save_cluster_results(results, output_file):
    """
    Save cluster results to a CSV file.
    """
    if not results:
        print("No results to save.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = [
        'Dataset', 'NClusters', 'MissingRate', 'Category', 
        'Winner_Method', 'Winner_ARI',
        'Precluster_Winner_Method', 'Precluster_Winner_ARI'
    ]
    results_df = results_df[column_order]
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"\nCluster results saved to: {output_file}")
    
    # Display summary
    print("\nSummary of cluster results:")
    print(results_df.to_string(index=False))

def main():
    # Folder containing the cluster datasets
    folder_path = 'zzzz.HPC Cluster Results'
    imputation_winners_file = 'imputation.append.csv'
    output_file = 'cluster.append.csv'
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Check if imputation results file exists
    if not os.path.exists(imputation_winners_file):
        print(f"Warning: Imputation results file '{imputation_winners_file}' not found.")
        print("Precluster winner information will not be available.")
    
    # Process all cluster datasets
    results = process_cluster_datasets(folder_path, imputation_winners_file)
    
    # Save results
    save_cluster_results(results, output_file)

if __name__ == "__main__":
    main()
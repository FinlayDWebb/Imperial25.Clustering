import pandas as pd
import os
import glob

def calculate_best_methods(df, dataset_name):
    """
    Calculate the best performing method for each MissingRate category and overall.
    
    Args:
        df (pandas DataFrame): Input data
        dataset_name (str): Name of the dataset
    
    Returns:
        list: List of dictionaries with winner information for each category
    """
    results = []
    
    # Get unique missing rates
    missing_rates = df['MissingRate'].unique()
    
    # Process each missing rate category
    for missing_rate in missing_rates:
        # Filter data for this missing rate
        df_subset = df[df['MissingRate'] == missing_rate]
        
        # Group by method and calculate average RMSE and PFC for this missing rate
        method_stats = df_subset.groupby('Method').agg({
            'RMSE': 'mean',
            'PFC': 'mean'
        }).reset_index()
        
        # Normalize RMSE (lower is better, so we use 1 - normalized)
        max_rmse = method_stats['RMSE'].max()
        min_rmse = method_stats['RMSE'].min()
        # Handle case where all RMSE values are the same
        if max_rmse == min_rmse:
            method_stats['Normalized_RMSE'] = 1.0
        else:
            method_stats['Normalized_RMSE'] = 1 - ((method_stats['RMSE'] - min_rmse) / (max_rmse - min_rmse))
        
        # Normalize PFC (lower is better, so we use 1 - normalized)
        max_pfc = method_stats['PFC'].max()
        min_pfc = method_stats['PFC'].min()
        # Handle case where all PFC values are the same
        if max_pfc == min_pfc:
            method_stats['Normalized_PFC'] = 1.0
        else:
            method_stats['Normalized_PFC'] = 1 - ((method_stats['PFC'] - min_pfc) / (max_pfc - min_pfc))
        
        # Calculate final score as average of normalized scores
        method_stats['Final_Score'] = (method_stats['Normalized_RMSE'] + method_stats['Normalized_PFC']) / 2
        
        # Find the winner for this missing rate
        winner_idx = method_stats['Final_Score'].idxmax()
        winner = method_stats.loc[winner_idx]
        
        results.append({
            'Dataset': dataset_name,
            'MissingRate': missing_rate,
            'Winner_Method': winner['Method'],
            'Winner_Score': winner['Final_Score'],
            'Avg_RMSE': winner['RMSE'],
            'Avg_PFC': winner['PFC'],
            'Category': f'MissingRate_{missing_rate}'
        })
    
    # Also calculate overall winner (across all missing rates)
    method_stats_overall = df.groupby('Method').agg({
        'RMSE': 'mean',
        'PFC': 'mean'
    }).reset_index()
    
    # Normalize RMSE for overall
    max_rmse = method_stats_overall['RMSE'].max()
    min_rmse = method_stats_overall['RMSE'].min()
    if max_rmse == min_rmse:
        method_stats_overall['Normalized_RMSE'] = 1.0
    else:
        method_stats_overall['Normalized_RMSE'] = 1 - ((method_stats_overall['RMSE'] - min_rmse) / (max_rmse - min_rmse))
    
    # Normalize PFC for overall
    max_pfc = method_stats_overall['PFC'].max()
    min_pfc = method_stats_overall['PFC'].min()
    if max_pfc == min_pfc:
        method_stats_overall['Normalized_PFC'] = 1.0
    else:
        method_stats_overall['Normalized_PFC'] = 1 - ((method_stats_overall['PFC'] - min_pfc) / (max_pfc - min_pfc))
    
    # Calculate final score for overall
    method_stats_overall['Final_Score'] = (method_stats_overall['Normalized_RMSE'] + method_stats_overall['Normalized_PFC']) / 2
    
    # Find the overall winner
    winner_idx = method_stats_overall['Final_Score'].idxmax()
    winner = method_stats_overall.loc[winner_idx]
    
    results.append({
        'Dataset': dataset_name,
        'MissingRate': 'Overall',
        'Winner_Method': winner['Method'],
        'Winner_Score': winner['Final_Score'],
        'Avg_RMSE': winner['RMSE'],
        'Avg_PFC': winner['PFC'],
        'Category': 'Overall'
    })
    
    return results

def process_datasets(folder_path):
    """
    Process all datasets in the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing CSV files
    
    Returns:
        list: List of dictionaries with winner information for each dataset
    """
    results = []
    
    # Find all CSV files in the folder that start with 'Imp.'
    pattern = os.path.join(folder_path, 'Imp.*.csv')
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No CSV files found matching pattern 'Imp.*.csv' in {folder_path}")
        return results
    
    print(f"Found {len(csv_files)} datasets to process...")
    
    for file_path in csv_files:
        try:
            # Extract dataset name from filename (remove 'Imp.' and '.csv')
            filename = os.path.basename(file_path)
            dataset_name = filename.replace('Imp.', '').replace('.csv', '')
            
            print(f"Processing dataset: {dataset_name}")
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Calculate best methods for each category
            dataset_results = calculate_best_methods(df, dataset_name)
            results.extend(dataset_results)
            
            # Print summary for this dataset
            for result in dataset_results:
                if result['Category'] == 'Overall':
                    print(f"  Overall Winner: {result['Winner_Method']} (Score: {result['Winner_Score']:.4f})")
                else:
                    print(f"  MissingRate {result['MissingRate']}: {result['Winner_Method']} (Score: {result['Winner_Score']:.4f})")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    return results

def save_results(results, output_file):
    """
    Save results to a CSV file.
    
    Args:
        results (list): List of result dictionaries
        output_file (str): Path to output CSV file
    """
    if not results:
        print("No results to save.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = ['Dataset', 'MissingRate', 'Category', 'Winner_Method', 'Winner_Score', 'Avg_RMSE', 'Avg_PFC']
    results_df = results_df[column_order]
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Display summary
    print("\nSummary of results:")
    print(results_df.to_string(index=False))

def main():
    # Folder containing the datasets
    folder_path = 'zzzz.HPC Imputation Results'
    output_file = 'imputation.append.csv'
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Process all datasets
    results = process_datasets(folder_path)
    
    # Save results
    save_results(results, output_file)

if __name__ == "__main__":
    main()
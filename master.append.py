import pandas as pd
import os

def create_master_append_csv(cluster_results_file, imputation_results_file, output_file):
    """
    Create a master CSV file combining information from cluster and imputation results.
    
    Args:
        cluster_results_file (str): Path to cluster results CSV
        imputation_results_file (str): Path to imputation results CSV
        output_file (str): Path to output master CSV
    """
    # Load both CSV files
    cluster_df = pd.read_csv(cluster_results_file)
    imputation_df = pd.read_csv(imputation_results_file)
    
    # Filter imputation results to only include specific missing rate categories (not overall)
    imputation_df = imputation_df[imputation_df['Category'].str.startswith('MissingRate_')]
    
    # Extract missing rate from Category column in imputation data
    imputation_df['MissingRate'] = imputation_df['Category'].str.replace('MissingRate_', '').astype(float)
    
    # Merge the dataframes
    merged_df = pd.merge(
        cluster_df,
        imputation_df[['Dataset', 'MissingRate', 'Winner_Method', 'Winner_Score', 'Avg_RMSE', 'Avg_PFC']],
        on=['Dataset', 'MissingRate'],
        how='left',
        suffixes=('_cluster', '_imputation')
    )
    
    # Rename columns to match the desired output format
    merged_df.rename(columns={
        'Dataset': 'DatasetName',
        'MissingRate': 'Missingness',
        'NClusters': 'NumbClusters',
        'Winner_Method_imputation': 'PreClusterWinner',
        'Winner_Method_cluster': 'PostClusterWinner',
        'Winner_Score': 'PreClusterMetric',
        'Precluster_Winner_ARI': 'PreClusterARI',
        'Winner_ARI': 'PostClusterARI',
        'Avg_RMSE': 'Avg_RMSE',
        'Avg_PFC': 'Avg_PFC'
    }, inplace=True)
    
    # Select only the required columns
    final_columns = [
        'DatasetName', 'Missingness', 'NumbClusters', 
        'PreClusterWinner', 'PostClusterWinner', 'PreClusterMetric',
        'PreClusterARI', 'PostClusterARI', 'Avg_RMSE', 'Avg_PFC'
    ]
    
    # Only include columns that exist in the dataframe
    available_columns = [col for col in final_columns if col in merged_df.columns]
    final_df = merged_df[available_columns]
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"Master CSV saved to: {output_file}")
    
    # Display summary
    print(f"\nCreated master CSV with {len(final_df)} rows")
    print("\nFirst few rows:")
    print(final_df.head().to_string(index=False))
    
    return final_df

def main():
    # File paths
    cluster_results_file = 'cluster.append.csv'
    imputation_results_file = 'imputation.append.csv'
    output_file = 'master.append.csv'
    
    # Check if input files exist
    if not os.path.exists(cluster_results_file):
        print(f"Error: Cluster results file '{cluster_results_file}' not found.")
        return
    
    if not os.path.exists(imputation_results_file):
        print(f"Error: Imputation results file '{imputation_results_file}' not found.")
        return
    
    # Create the master CSV
    master_df = create_master_append_csv(cluster_results_file, imputation_results_file, output_file)

if __name__ == "__main__":
    main()
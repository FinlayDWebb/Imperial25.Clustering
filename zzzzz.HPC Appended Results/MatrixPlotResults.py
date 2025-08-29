import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set Helvetica font
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

def create_divergence_matrix(csv_file='master.append.csv', figsize=(16, 12), dpi=300):
    """
    Create a sophisticated divergence matrix visualization for clustering performance analysis.
    
    Parameters:
    csv_file (str): Path to the CSV file with clustering results
    figsize (tuple): Figure size in inches
    dpi (int): Resolution for high-quality output
    """
    
    # Load and process data
    df = pd.read_csv(csv_file)
    
    # Calculate ΔARI
    df['DeltaARI'] = df['PostClusterARI'] - df['PreClusterARI']
    df['IsDifferentWinner'] = df['PreClusterWinner'] != df['PostClusterWinner']
    
    # Get unique values for matrix dimensions
    missingness_rates = sorted(df['Missingness'].unique())
    cluster_numbers = sorted(df['NumbClusters'].unique())
    
    # Custom dataset ordering: push Manila and Weather to bottom
    all_datasets = sorted(df['DatasetName'].unique())
    priority_datasets = [d for d in all_datasets if d.lower() not in ['manila', 'weather']]
    bottom_datasets = [d for d in all_datasets if d.lower() in ['manila', 'weather']]
    datasets = priority_datasets + bottom_datasets
    
    # Set up the figure with high quality settings
    plt.style.use('default')
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('#fafafa')
    
    # Calculate grid dimensions
    n_rows = len(missingness_rates)
    n_cols = len(cluster_numbers)
    
    # Create grid layout with space for labels - shared y-axis on left only
    gs = fig.add_gridspec(n_rows + 1, n_cols + 1, 
                         height_ratios=[0.3] + [1] * n_rows,
                         width_ratios=[0.4] + [1] * n_cols,
                         hspace=0.3, wspace=0.15,
                         left=0.08, right=0.95, top=0.88, bottom=0.12)
    
    
    # Add column headers
    for j, k in enumerate(cluster_numbers):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.text(0.5, 0.5, f'k = {int(k)}', ha='center', va='center', 
                fontsize=14, fontweight='bold', color='#34495e',
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Add row headers
    for i, miss_rate in enumerate(missingness_rates):
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.text(0.5, 0.5, f'{int(miss_rate * 100)}%\nMissing', ha='center', va='center',
                fontsize=12, fontweight='bold', color='#34495e',
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Create individual lollipop charts
    for i, miss_rate in enumerate(missingness_rates):
        for j, k in enumerate(cluster_numbers):
            ax = fig.add_subplot(gs[i + 1, j + 1])
            
            # Filter data for this cell
            cell_data = df[(df['Missingness'] == miss_rate) & 
                          (df['NumbClusters'] == k)].copy()
            
            # Create full dataset list with proper ordering for this cell
            cell_datasets_present = set(cell_data['DatasetName']) if len(cell_data) > 0 else set()
            
            # Calculate global max for consistent scaling
            global_max_delta = df['DeltaARI'].abs().max() if len(df) > 0 else 0.1
            x_limit = global_max_delta * 1.15
            
            # Y-axis setup - all datasets in order, with missing ones as grey placeholders
            y_positions = {dataset: idx for idx, dataset in enumerate(datasets)}
            
            # Color function with brighter greens
            def get_color(delta_ari, is_different):
                if is_different:  # Different winner - red tones
                    intensity = min(abs(delta_ari) / global_max_delta, 1.0)
                    return plt.cm.Reds(0.3 + intensity * 0.5)
                else:  # Same winner - brighter green tones
                    intensity = min(abs(delta_ari) / global_max_delta, 1.0)
                    # Use a brighter green colormap range
                    return plt.cm.Greens(0.4 + intensity * 0.6)
            
            # Draw all datasets, with grey markers for missing data
            for dataset in datasets:
                y_pos = y_positions[dataset]
                
                if dataset in cell_datasets_present:
                    # Draw actual data
                    row = cell_data[cell_data['DatasetName'] == dataset].iloc[0]
                    delta = row['DeltaARI']
                    is_diff = row['IsDifferentWinner']
                    color = get_color(delta, is_diff)
                    
                    # Draw stem (line from 0 to value)
                    ax.plot([0, delta], [y_pos, y_pos], 
                           color=color, linewidth=2.5, alpha=0.8)
                    
                    # Draw head (dot at end)
                    ax.scatter([delta], [y_pos], 
                              color=color, s=40, zorder=3,
                              edgecolors='white', linewidth=1)
                else:
                    # Draw grey placeholder for missing data
                    ax.scatter([0], [y_pos], 
                              color='lightgrey', s=25, zorder=2, alpha=0.5,
                              edgecolors='grey', linewidth=0.5)
            
            # Draw zero line
            ax.axvline(x=0, color='#333', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Formatting
            ax.set_xlim(-x_limit, x_limit)
            ax.set_ylim(-0.5, len(datasets) - 0.5)
            
            # Y-axis (dataset names) - only show on leftmost column
            ax.set_yticks(list(y_positions.values()))
            if j == 0:  # Only leftmost column gets y-labels
                ax.set_yticklabels(datasets, fontsize=9)
            else:
                ax.set_yticklabels([])
            
            # X-axis
            ax.tick_params(axis='x', labelsize=8)
            if i == len(missingness_rates) - 1:  # Only show x-label on bottom row
                ax.set_xlabel('ΔARI', fontsize=10, color='#555')
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#ddd')
            ax.spines['bottom'].set_color('#ddd')
            
            # Grid
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Add cell title
            ax.text(0.02, 0.98, f'{int(miss_rate*100)}%, k={int(k)}',
                   transform=ax.transAxes, fontsize=8, fontweight='bold',
                   color='#7f8c8d', verticalalignment='top')
    
    # Add axis labels
    fig.text(0.05, 0.5, 'Missingness Rate', rotation=90, ha='center', va='center',
             fontsize=16, fontweight='bold', color="#000000")
    fig.text(0.5, 0.90, 'Number of Clusters (k)', ha='center', va='center',
             fontsize=16, fontweight='bold', color="#000000")
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color=plt.cm.Reds(0.6), label='Positive ΔARI (Brittle Imputation)'),
        mpatches.Patch(color=plt.cm.Greens(0.7), label='Null ΔARI (Robust Imputation)'),
        mpatches.Patch(color='lightgrey', label='Missing Data'),
        mpatches.Patch(color='#333', label='Zero Line (PreCluster Winner)')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
              fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    return fig

def save_high_quality_figure(fig, filename='clustering_divergence_matrix', formats=['png', 'pdf']):
    """
    Save the figure in multiple high-quality formats suitable for academic posters.
    """
    for fmt in formats:
        fig.savefig(f'{filename}.{fmt}', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved: {filename}.{fmt}")

# Example usage
if __name__ == "__main__":
    # Create the visualization
    fig = create_divergence_matrix('master.append.csv')
    
    # Save in high quality formats
    save_high_quality_figure(fig, 'imperial_clustering_divergence')
    
    # Display the plot
    plt.show()

# Additional utility function to explore your data structure
def analyze_data_structure(csv_file='master.append.csv'):
    """
    Analyze the structure of your data to ensure proper visualization setup.
    """
    df = pd.read_csv(csv_file)
    
    print("Data Overview:")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nUnique missingness rates: {sorted(df['Missingness'].unique())}")
    print(f"Unique cluster numbers: {sorted(df['NumbClusters'].unique())}")
    print(f"Unique datasets: {sorted(df['DatasetName'].unique())}")
    print(f"\nMethods found:")
    print(f"PreCluster winners: {df['PreClusterWinner'].unique()}")
    print(f"PostCluster winners: {df['PostClusterWinner'].unique()}")
    
    # Check for divergence cases
    divergence_cases = df[df['PreClusterWinner'] != df['PostClusterWinner']]
    print(f"\nDivergence cases (different winners): {len(divergence_cases)}/{len(df)} ({len(divergence_cases)/len(df)*100:.1f}%)")
    
    return df

# Uncomment to analyze your data first:
# analyze_data_structure('master.append.csv')
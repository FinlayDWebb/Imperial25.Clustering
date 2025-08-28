import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches

# Set style for clean, academic look
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Generate synthetic clustered data
np.random.seed(42)
n_samples = 150
n_centers = 3
X, y = make_blobs(n_samples=n_samples, centers=n_centers, n_features=2, 
                  random_state=42, cluster_std=0.8, center_box=(-3.0, 3.0))

# Define colors for clusters
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
cluster_colors = [colors[label] for label in y]

# Create figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('The Illusion of Accuracy: Why RMSE Isn\'t Everything', 
             fontsize=20, fontweight='bold', y=0.95)

# Function to draw cluster boundaries (convex hulls)
from scipy.spatial import ConvexHull

def draw_cluster_boundaries(ax, X, y, alpha=0.2):
    for cluster_id in np.unique(y):
        cluster_points = X[y == cluster_id]
        if len(cluster_points) > 2:
            hull = ConvexHull(cluster_points)
            for simplex in hull.simplices:
                ax.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 
                       'k-', alpha=0.3, linewidth=1)
            # Fill the hull
            hull_points = cluster_points[hull.vertices]
            ax.fill(hull_points[:, 0], hull_points[:, 1], 
                   color=colors[cluster_id], alpha=alpha)

# Function to draw connections within clusters
def draw_intracluster_connections(ax, X, y, max_connections=3):
    for cluster_id in np.unique(y):
        cluster_points = X[y == cluster_id]
        cluster_indices = np.where(y == cluster_id)[0]
        
        if len(cluster_points) > 1:
            # Find nearest neighbors within cluster
            nbrs = NearestNeighbors(n_neighbors=min(max_connections+1, len(cluster_points)))
            nbrs.fit(cluster_points)
            distances, indices = nbrs.kneighbors(cluster_points)
            
            for i, point_indices in enumerate(indices):
                for j in point_indices[1:]:  # Skip self (index 0)
                    ax.plot([cluster_points[i, 0], cluster_points[j, 0]], 
                           [cluster_points[i, 1], cluster_points[j, 1]], 
                           'gray', alpha=0.4, linewidth=0.8, zorder=1)

# Panel 1: Ground Truth
ax1 = axes[0]
draw_cluster_boundaries(ax1, X, y, alpha=0.15)
draw_intracluster_connections(ax1, X, y)

scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=cluster_colors, s=60, alpha=0.8, 
                      edgecolors='white', linewidth=1, zorder=3)

ax1.set_title('Panel 1: Ground Truth', fontsize=16, fontweight='bold', pad=20)
ax1.text(0.5, -0.12, 'Original data with clear\ncluster structure preserved', 
         transform=ax1.transAxes, ha='center', va='top', fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
ax1.set_xlabel('Feature 1', fontsize=12)
ax1.set_ylabel('Feature 2', fontsize=12)
ax1.grid(True, alpha=0.3)

# Panel 2: Brittle Imputation - exaggerated structure destruction with low RMSE
np.random.seed(123)

# Create a more dramatic structure-destroying transformation while keeping RMSE low
X_brittle = X.copy()

# For each point, move it towards the nearest point from a DIFFERENT cluster
# This creates maximum structural damage while keeping movements relatively small
for i in range(len(X)):
    current_cluster = y[i]
    current_point = X[i]
    
    # Find nearest points from different clusters
    other_cluster_points = []
    other_cluster_indices = []
    
    for j in range(len(X)):
        if y[j] != current_cluster:
            other_cluster_points.append(X[j])
            other_cluster_indices.append(j)
    
    if other_cluster_points:
        other_cluster_points = np.array(other_cluster_points)
        # Find the closest point from another cluster
        distances = np.linalg.norm(other_cluster_points - current_point, axis=1)
        closest_other_idx = np.argmin(distances)
        closest_other_point = other_cluster_points[closest_other_idx]
        
        # Move current point 25% towards the closest point from another cluster
        # This keeps RMSE relatively low but maximally destroys cluster structure
        move_fraction = 0.25
        movement = (closest_other_point - current_point) * move_fraction
        X_brittle[i] = current_point + movement

ax2 = axes[1]
# Draw original boundaries (faded) to show distortion
draw_cluster_boundaries(ax2, X, y, alpha=0.05)

# Draw distorted connections - these will look very chaotic now
draw_intracluster_connections(ax2, X_brittle, y)

scatter2 = ax2.scatter(X_brittle[:, 0], X_brittle[:, 1], c=cluster_colors, s=60, alpha=0.8,
                      edgecolors='white', linewidth=1, zorder=3)

ax2.set_title('Panel 2: Brittle Imputation', fontsize=16, fontweight='bold', pad=20)
ax2.text(0.5, -0.12, 'Good RMSE, but cluster\nboundaries are blurred', 
         transform=ax2.transAxes, ha='center', va='top', fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
ax2.set_xlabel('Feature 1', fontsize=12)
ax2.set_ylabel('Feature 2', fontsize=12)
ax2.grid(True, alpha=0.3)

# Panel 3: Robust Imputation (structure-preserving transformation)
X_robust = X.copy()
# Apply small structure-preserving transformations to each cluster
for cluster_id in np.unique(y):
    cluster_mask = y == cluster_id
    cluster_points = X[cluster_mask]
    cluster_center = np.mean(cluster_points, axis=0)
    
    # Apply small rotation and scaling within cluster
    angle = np.random.uniform(-0.1, 0.1)  # Small rotation
    scale = np.random.uniform(0.95, 1.05)  # Small scaling
    
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    # Center, transform, and restore
    centered_points = cluster_points - cluster_center
    transformed_points = (centered_points @ rotation_matrix.T) * scale
    X_robust[cluster_mask] = transformed_points + cluster_center

ax3 = axes[2]
draw_cluster_boundaries(ax3, X_robust, y, alpha=0.15)
draw_intracluster_connections(ax3, X_robust, y)

scatter3 = ax3.scatter(X_robust[:, 0], X_robust[:, 1], c=cluster_colors, s=60, alpha=0.8,
                      edgecolors='white', linewidth=1, zorder=3)

ax3.set_title('Panel 3: Robust Imputation', fontsize=16, fontweight='bold', pad=20)
ax3.text(0.5, -0.12, 'Good RMSE AND preserved\ncluster structure', 
         transform=ax3.transAxes, ha='center', va='top', fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
ax3.set_xlabel('Feature 1', fontsize=12)
ax3.set_ylabel('Feature 2', fontsize=12)
ax3.grid(True, alpha=0.3)

# Calculate and display RMSE for each method
rmse_brittle = np.sqrt(np.mean((X - X_brittle)**2))
rmse_robust = np.sqrt(np.mean((X - X_robust)**2))

# Add RMSE annotations
ax2.text(0.02, 0.98, f'RMSE: {rmse_brittle:.3f}', transform=ax2.transAxes, 
         fontsize=11, va='top', ha='left', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
ax3.text(0.02, 0.98, f'RMSE: {rmse_robust:.3f}', transform=ax3.transAxes, 
         fontsize=11, va='top', ha='left',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# Add legend
legend_elements = [mpatches.Patch(color=colors[i], label=f'Cluster {i+1}') for i in range(3)]
legend_elements.append(mpatches.Patch(color='gray', alpha=0.4, label='Intra-cluster connections'))

fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
          ncol=4, frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.15)

# Add main message
fig.text(0.5, 0.08, 'Key Insight: Traditional metrics like RMSE can mask structural damage to data relationships',
         ha='center', va='center', fontsize=14, style='italic', 
         bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.3))

plt.show()

# Print RMSE values for reference
print(f"RMSE Comparison:")
print(f"Brittle method: {rmse_brittle:.4f}")
print(f"Robust method: {rmse_robust:.4f}")
print(f"Difference: {abs(rmse_brittle - rmse_robust):.4f}")
print("\nDespite similar RMSE values, the structural preservation is dramatically different!")
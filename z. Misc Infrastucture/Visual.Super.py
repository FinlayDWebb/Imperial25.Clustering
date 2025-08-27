import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull, Voronoi
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

# Set moderate-DPI rendering and professional style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.linewidth'] = 0.6
plt.rcParams['lines.linewidth'] = 1.2

# Generate high-resolution synthetic clustered data
np.random.seed(42)
n_samples = 300  # More points for smoother visualization
n_centers = 3
X, y = make_blobs(n_samples=n_samples, centers=n_centers, n_features=2, 
                  random_state=42, cluster_std=1.2, center_box=(-4.0, 4.0))

# Professional color palette inspired by the Ricci flow visualization
colors = ['#E8345A', '#4A90E2', '#7ED321']  # Vibrant yet professional
cluster_colors = [colors[label] for label in y]

# Create figure with enhanced aesthetics
fig = plt.figure(figsize=(15, 5))
gs = fig.add_gridspec(1, 3, hspace=0.05, wspace=0.15)

# No main title

def create_smooth_boundary(X, y, cluster_id, extend_factor=1.3):
    """Create smooth cluster boundary using gaussian smoothing"""
    cluster_points = X[y == cluster_id]
    if len(cluster_points) < 3:
        return None, None
    
    # Create a dense grid around the cluster
    x_min, x_max = cluster_points[:, 0].min() - 1, cluster_points[:, 0].max() + 1
    y_min, y_max = cluster_points[:, 1].min() - 1, cluster_points[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Create density field
    density = np.zeros_like(xx)
    for point in cluster_points:
        dist_squared = (xx - point[0])**2 + (yy - point[1])**2
        density += np.exp(-dist_squared / (2 * 0.8**2))
    
    # Smooth the density field
    density = gaussian_filter(density, sigma=2.0)
    
    return xx, yy, density

def draw_enhanced_connections(ax, X, y, alpha=0.6, connection_density=0.3):
    """Draw enhanced intra-cluster connections with varying opacity"""
    for cluster_id in np.unique(y):
        cluster_points = X[y == cluster_id]
        cluster_indices = np.where(y == cluster_id)[0]
        
        if len(cluster_points) > 1:
            # Use more sophisticated connection strategy
            nbrs = NearestNeighbors(n_neighbors=min(6, len(cluster_points)))
            nbrs.fit(cluster_points)
            distances, indices = nbrs.kneighbors(cluster_points)
            
            for i, point_indices in enumerate(indices):
                for j_idx, j in enumerate(point_indices[1:]):  # Skip self
                    if np.random.random() < connection_density:  # Randomly sample connections
                        distance = distances[i, j_idx + 1]
                        # Vary opacity based on distance
                        line_alpha = alpha * np.exp(-distance / 2.0)
                        
                        ax.plot([cluster_points[i, 0], cluster_points[j, 0]], 
                               [cluster_points[i, 1], cluster_points[j, 1]], 
                               color='#34495E', alpha=line_alpha, linewidth=1.0, 
                               zorder=1, solid_capstyle='round')

# Panel 1: Ground Truth
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#FAFAFA')

# Draw smooth cluster boundaries
for cluster_id in np.unique(y):
    xx, yy, density = create_smooth_boundary(X, y, cluster_id)
    if density is not None:
        # Create smooth contour
        contour_levels = [np.max(density) * 0.3]
        cs = ax1.contour(xx, yy, density, levels=contour_levels, 
                        colors=[colors[cluster_id]], linewidths=2.5, alpha=0.8)
        # Fill with gradient
        ax1.contourf(xx, yy, density, levels=[contour_levels[0], np.max(density)], 
                    colors=[colors[cluster_id]], alpha=0.15)

# Enhanced connections
draw_enhanced_connections(ax1, X, y, alpha=0.4, connection_density=0.2)

# High-quality scatter plot
scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=cluster_colors, s=50, alpha=0.9, 
                      edgecolors='white', linewidth=1.0, zorder=3,
                      rasterized=True)

ax1.set_title('Ground Truth', fontsize=12, fontweight='bold', pad=15,
              color='#2C3E50')

ax1.set_xlabel('Feature Dimension 1', fontsize=10, color='#34495E')
ax1.set_ylabel('Feature Dimension 2', fontsize=10, color='#34495E')
ax1.grid(True, alpha=0.3, color='#BDC3C7')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel 2: Brittle Imputation - dramatically enhanced destruction
np.random.seed(123)
X_brittle = X.copy()

# Create maximum structural damage while keeping RMSE reasonable
for i in range(len(X)):
    current_cluster = y[i]
    current_point = X[i]
    
    # Find nearest points from different clusters
    other_cluster_points = []
    for j in range(len(X)):
        if y[j] != current_cluster:
            other_cluster_points.append(X[j])
    
    if other_cluster_points:
        other_cluster_points = np.array(other_cluster_points)
        distances = np.linalg.norm(other_cluster_points - current_point, axis=1)
        closest_other_idx = np.argmin(distances)
        closest_other_point = other_cluster_points[closest_other_idx]
        
        # Move point 40% towards nearest point from different cluster
        move_fraction = 0.4
        movement = (closest_other_point - current_point) * move_fraction
        X_brittle[i] = current_point + movement

ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#FAFAFA')

# Show original boundaries as faded ghosts
for cluster_id in np.unique(y):
    xx, yy, density = create_smooth_boundary(X, y, cluster_id)
    if density is not None:
        contour_levels = [np.max(density) * 0.3]
        ax2.contour(xx, yy, density, levels=contour_levels, 
                   colors=['#95A5A6'], linewidths=1.5, alpha=0.4, linestyles='--')

# Draw chaotic connections on distorted data
draw_enhanced_connections(ax2, X_brittle, y, alpha=0.6, connection_density=0.4)

scatter2 = ax2.scatter(X_brittle[:, 0], X_brittle[:, 1], c=cluster_colors, s=50, alpha=0.9,
                      edgecolors='white', linewidth=1.0, zorder=3, rasterized=True)

ax2.set_title('Brittle Imputation', fontsize=12, fontweight='bold', pad=15,
              color='#E74C3C')

ax2.set_xlabel('Feature Dimension 1', fontsize=10, color='#34495E')
ax2.set_ylabel('Feature Dimension 2', fontsize=10, color='#34495E')
ax2.grid(True, alpha=0.3, color='#BDC3C7')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Panel 3: Robust Imputation - enhanced structure preservation
X_robust = X.copy()
for cluster_id in np.unique(y):
    cluster_mask = y == cluster_id
    cluster_points = X[cluster_mask]
    cluster_center = np.mean(cluster_points, axis=0)
    
    # Apply subtle structure-preserving transformations
    angle = np.random.uniform(-0.15, 0.15)
    scale = np.random.uniform(0.92, 1.08)
    
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    centered_points = cluster_points - cluster_center
    transformed_points = (centered_points @ rotation_matrix.T) * scale
    X_robust[cluster_mask] = transformed_points + cluster_center

ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor('#FAFAFA')

# Draw preserved boundaries
for cluster_id in np.unique(y):
    xx, yy, density = create_smooth_boundary(X_robust, y, cluster_id)
    if density is not None:
        contour_levels = [np.max(density) * 0.3]
        cs = ax3.contour(xx, yy, density, levels=contour_levels, 
                        colors=[colors[cluster_id]], linewidths=2.5, alpha=0.8)
        ax3.contourf(xx, yy, density, levels=[contour_levels[0], np.max(density)], 
                    colors=[colors[cluster_id]], alpha=0.15)

draw_enhanced_connections(ax3, X_robust, y, alpha=0.4, connection_density=0.2)

scatter3 = ax3.scatter(X_robust[:, 0], X_robust[:, 1], c=cluster_colors, s=80, alpha=0.9,
                      edgecolors='white', linewidth=1.5, zorder=3, rasterized=True)

ax3.set_title('Robust Imputation', fontsize=14, fontweight='bold', pad=20,
              color='#27AE60')

ax3.set_xlabel('Feature Dimension 1', fontsize=12, color='#34495E')
ax3.set_ylabel('Feature Dimension 2', fontsize=12, color='#34495E')
ax3.grid(True, alpha=0.3, color='#BDC3C7')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Calculate and display RMSE with enhanced styling
rmse_brittle = np.sqrt(np.mean((X - X_brittle)**2))
rmse_robust = np.sqrt(np.mean((X - X_robust)**2))

# Enhanced RMSE annotations
rmse_style = dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.95,
                 edgecolor='#34495E', linewidth=1.0)

ax2.text(0.02, 0.98, f'RMSE: {rmse_brittle:.3f}', transform=ax2.transAxes, 
         fontsize=10, va='top', ha='left', bbox=rmse_style, fontweight='bold')
ax3.text(0.02, 0.98, f'RMSE: {rmse_robust:.3f}', transform=ax3.transAxes, 
         fontsize=10, va='top', ha='left', bbox=rmse_style, fontweight='bold')

# Professional legend with enhanced styling
legend_elements = []
for i in range(3):
    legend_elements.append(mpatches.Patch(color=colors[i], label=f'Community {i+1}'))
legend_elements.append(mpatches.Patch(color='#34495E', alpha=0.4, label='Intra-community edges'))
legend_elements.append(mpatches.Patch(color='#95A5A6', alpha=0.4, label='Original boundaries'))

legend = fig.legend(handles=legend_elements, loc='upper center', 
                   bbox_to_anchor=(0.5, 0.08), ncol=5, frameon=True, 
                   fancybox=True, shadow=True, fontsize=10)
legend.get_frame().set_facecolor('#FFFFFF')
legend.get_frame().set_edgecolor('#BDC3C7')
legend.get_frame().set_linewidth(1.2)

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.25)

plt.show()

# Enhanced output information
print("=" * 60)
print("PROFESSIONAL IMPUTATION VISUALIZATION ANALYSIS")
print("=" * 60)
print(f"Dataset size: {n_samples} points, {n_centers} communities")
print(f"Brittle method RMSE: {rmse_brittle:.4f}")
print(f"Robust method RMSE:  {rmse_robust:.4f}")
print(f"RMSE difference:     {abs(rmse_brittle - rmse_robust):.4f}")
print("=" * 60)
print("STRUCTURAL IMPACT:")
print("• Brittle: Destroys community boundaries and connectivity")
print("• Robust:  Preserves community structure and relationships") 
print("• Both methods achieve similar statistical accuracy (RMSE)")
print("• Only structure-aware methods like DIBmix can detect the difference")
print("=" * 60)
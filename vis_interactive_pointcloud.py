"""
Interactive 3D point cloud visualization script using Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

def visualize_pointcloud_interactive(xyz_cView, save_path=None):
    """
    Create an interactive 3D point cloud visualization that can be rotated by dragging.
    
    Args:
        xyz_cView: torch.Tensor of shape [N, H, W, 3] or [H, W, 3] containing (x, y, z) coordinates
        save_path: Optional path to save the figure. If None, only displays interactively.
    
    Usage:
        - Left click + drag: Rotate the view
        - Right click + drag: Pan the view
        - Scroll: Zoom in/out
        - Close window to continue
    """
    # Handle torch tensors
    if torch.is_tensor(xyz_cView):
        xyz = xyz_cView.detach().cpu().numpy()
    else:
        xyz = np.array(xyz_cView)
    
    # Handle batch dimension
    if xyz.ndim == 4:
        xyz = xyz[0]  # Take first sample
    
    # Get dimensions
    H, W, _ = xyz.shape
    
    # Reshape to point cloud
    x = xyz[:, :, 0].flatten()
    y = xyz[:, :, 1].flatten()
    z = xyz[:, :, 2].flatten()
    
    # Remove invalid points (NaN, Inf)
    valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]
    
    print(f"Valid points: {len(x)} / {H*W}")
    
    # Subsample if too many points (for faster rendering)
    max_points = 50000
    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False)
        x = x[indices]
        y = y[indices]
        z = z[indices]
        print(f"Subsampled to {max_points} points for performance")
    
    # Color by depth (z-coordinate)
    # Use percentile clipping for better visualization
    vmin = np.percentile(z, 5)
    vmax = np.percentile(z, 95)
    z_normalized = np.clip((z - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    
    # Apply gamma correction for darker colors
    gamma = 1.4
    z_normalized = z_normalized ** gamma
    
    # Create figure with dark background
    fig = plt.figure(figsize=(12, 9), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Create colormap (viridis or jet)
    cmap = plt.get_cmap('viridis')
    colors = cmap(z_normalized)[:, :3] * 0.7  # Darken colors
    
    # Plot point cloud
    scatter = ax.scatter(x, y, z, c=colors, s=1, alpha=0.8)
    
    # Set labels with white color
    ax.set_xlabel('X (mm)', color='white', fontsize=10)
    ax.set_ylabel('Y (mm)', color='white', fontsize=10)
    ax.set_zlabel('Z (mm)', color='white', fontsize=10)
    ax.set_title('Interactive 3D Point Cloud\n(Drag to rotate, scroll to zoom)', 
                 color='white', fontsize=12, pad=20)
    
    # Set tick colors to white
    ax.tick_params(colors='white', labelsize=8)
    
    # Set pane colors to dark
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.grid(True, color='gray', alpha=0.3)
    
    # Set equal aspect ratio for better visualization
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add colorbar
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(z)
    mappable.set_clim(vmin, vmax)
    cbar = plt.colorbar(mappable, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Depth Z (mm)', color='white', fontsize=10)
    cbar.ax.tick_params(colors='white', labelsize=8)
    
    # Print instructions
    print("\n" + "="*60)
    print("Interactive Controls:")
    print("  - Left click + drag: Rotate the view")
    print("  - Right click + drag: Pan the view")
    print("  - Scroll wheel: Zoom in/out")
    print("  - Close window to continue")
    print("="*60 + "\n")
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', edgecolor='none', bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    # Show interactive plot
    plt.show()
    
    return fig, ax


def visualize_from_debug_console():
    """
    Convenience function to use in debug console.
    Automatically detects xyz_cView variable.
    """
    import sys
    
    # Get the calling frame's local variables
    frame = sys._getframe(1)
    local_vars = frame.f_locals
    
    # Try to find xyz_cView variable
    if 'xyz_cView' in local_vars:
        xyz = local_vars['xyz_cView']
        print("Found xyz_cView in current scope")
    elif 'xyz_cView_crop_hat' in local_vars:
        xyz = local_vars['xyz_cView_crop_hat']
        print("Found xyz_cView_crop_hat in current scope")
    else:
        print("Error: Could not find xyz_cView or xyz_cView_crop_hat variable")
        print(f"Available variables: {list(local_vars.keys())}")
        return None
    
    # Visualize
    save_path = './vis_debug_pattern/xyz_cView_interactive.png'
    return visualize_pointcloud_interactive(xyz, save_path=save_path)


if __name__ == "__main__":
    print("This script is designed to be used in the Python Debug Console.")
    print("\nUsage:")
    print("  1. Import: from vis_interactive_pointcloud import visualize_pointcloud_interactive")
    print("  2. Call: visualize_pointcloud_interactive(xyz_cView)")
    print("\nOr simply:")
    print("  exec(open('vis_interactive_pointcloud.py').read())")
    print("  visualize_from_debug_console()")

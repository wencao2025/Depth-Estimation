"""
Interactive 3D point cloud viewer using Plotly.
Produces a standalone HTML file you can open in any browser for full drag/zoom interaction.

Usage:
    from vis_interactive_pointcloud_plotly import visualize_pointcloud_plotly
    visualize_pointcloud_plotly(xyz_cView, out_html='pc_view.html')

If `plotly` is not installed, install with:
    pip install plotly

This script works on headless servers because it only writes an HTML file.
"""

import numpy as np
import os
import webbrowser
import torch

try:
    import plotly.graph_objects as go
    from plotly.offline import plot
except Exception:
    go = None
    plot = None


def visualize_pointcloud_plotly(xyz_cView, out_html="./vis_debug_pattern/xyz_cView_plotly.html",
                                max_points=200000, auto_open=True, colormap="Viridis"):
    """
    Create an interactive 3D point cloud HTML (Plotly) that can be opened in a browser.

    Args:
        xyz_cView: torch.Tensor or numpy array with shape [N, H, W, 3] or [H, W, 3].
        out_html: path to output .html file. Directory will be created if needed.
        max_points: maximum number of points to include (subsamples if larger).
        auto_open: whether to open the generated HTML in the default browser.
        colormap: Plotly colorscale name (e.g., 'Viridis', 'Cividis', 'Turbo')

    Returns:
        out_html: path to written HTML file, or None if plotly is not available.
    """
    if go is None or plot is None:
        print("Plotly is not available. Install it with `pip install plotly`.")
        return None

    # Convert tensor to numpy
    if torch.is_tensor(xyz_cView):
        xyz = xyz_cView.detach().cpu().numpy()
    else:
        xyz = np.array(xyz_cView)

    # Handle batch
    if xyz.ndim == 4:
        xyz = xyz[0]

    # Reshape to (N,3)
    H, W, C = xyz.shape
    assert C >= 3, "xyz_cView must have at least 3 channels"
    pts = xyz.reshape(-1, C)[:, :3]

    # Remove invalid
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]

    n_points = pts.shape[0]
    if n_points == 0:
        print("No valid points to plot.")
        return None

    # Subsample
    if n_points > max_points:
        idx = np.random.choice(n_points, max_points, replace=False)
        pts = pts[idx]
        n_points = max_points
        print(f"Subsampled to {max_points} points for browser performance")

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    # Color by depth (z)
    vmin = np.percentile(z, 5)
    vmax = np.percentile(z, 95)
    zc = np.clip((z - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)

    # Create scatter3d
    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=zc,
            colorscale=colormap,
            opacity=0.8,
            colorbar=dict(title='Normalized depth')
        )
    )

    layout = go.Layout(
        title='Interactive 3D Point Cloud (Plotly)\n(Drag to rotate, scroll to zoom)',
        scene=dict(
            xaxis=dict(title='X', backgroundcolor='black', gridcolor='gray', showbackground=True, zerolinecolor='gray'),
            yaxis=dict(title='Y', backgroundcolor='black', gridcolor='gray', showbackground=True, zerolinecolor='gray'),
            zaxis=dict(title='Z', backgroundcolor='black', gridcolor='gray', showbackground=True, zerolinecolor='gray'),
            aspectmode='data'
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white')
    )

    fig = go.Figure(data=[scatter], layout=layout)

    # Ensure output dir
    out_dir = os.path.dirname(out_html)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Write standalone HTML
    plot(fig, filename=out_html, auto_open=False)
    print(f"Saved interactive HTML to: {out_html}")

    # Optionally open in browser (works on machine with GUI)
    if auto_open:
        try:
            webbrowser.open('file://' + os.path.abspath(out_html))
        except Exception as e:
            print(f"Could not auto-open browser: {e}")

    return out_html


if __name__ == '__main__':
    print('This script writes a standalone HTML interactive viewer for a point cloud.')
    print('Example usage:')
    print("  from vis_interactive_pointcloud_plotly import visualize_pointcloud_plotly")
    print("  visualize_pointcloud_plotly(xyz_cView, out_html='vis_debug_pattern/pc.html')")

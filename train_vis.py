"""
Visualization utilities for training - PyTorch version
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from matplotlib import cm


import webbrowser

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

def _normalize_for_visual(arr, mask=None, clip_percentile=(1, 99), eps=1e-8):
    """
    Normalize array to [0,1] for visualization.
    
    Args:
        arr: 2D numpy array
        mask: 2D boolean/0-1 mask same size (optional)
        clip_percentile: tuple (low, high) percentiles to clip extremes
        eps: small value to avoid division by zero
    
    Returns:
        Normalized array in [0, 1]
    """
    if mask is not None:
        valid = mask > 0
        if valid.sum() == 0:
            mn, mx = arr.min(), arr.max()
        else:
            vals = arr[valid]
            if len(vals) > 0:
                lo = np.percentile(vals, clip_percentile[0])
                hi = np.percentile(vals, clip_percentile[1])
                mn, mx = lo, hi
            else:
                mn, mx = arr.min(), arr.max()
    else:
        lo = np.percentile(arr, clip_percentile[0])
        hi = np.percentile(arr, clip_percentile[1])
        mn, mx = lo, hi
    
    # Avoid zero range
    if abs(mx - mn) < eps:
        mn = arr.min()
        mx = arr.max()
        if abs(mx - mn) < eps:
            return np.zeros_like(arr)
    
    out = (arr - mn) / (mx - mn)
    out = np.clip(out, 0.0, 1.0)
    return out


def to_numpy(tensor):
    """Convert torch tensor to numpy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def save_single_channel_image(arr, save_path, cmap='viridis', dpi=100):
    """
    Save a 2D array as an image with colormap.
    
    Args:
        arr: 2D numpy array
        save_path: output file path
        cmap: matplotlib colormap name
        dpi: resolution
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(arr, cmap=cmap)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def visualize_xyz_prediction(xyz_tensor, mask_tensor, outdir, prefix, idx, max_samples=3, save_to_disk=True, suffix='pred', apply_mask=True):
    """
    Visualize xyz prediction with x, y, z in a single figure with 3 subplots.
    
    Args:
        xyz_tensor: [B, H, W, 3] tensor (NHWC format)
        mask_tensor: [B, H, W, 1] or [B, H, W] mask tensor
        outdir: output directory
        prefix: filename prefix (e.g., 'train', 'valid')
        idx: iteration/step index
        max_samples: maximum number of samples to save from batch
        save_to_disk: whether to save to disk (filename without iter)
        suffix: suffix for filename ('pred' or 'gt')
    
    Returns:
        Dictionary of images for TensorBoard (CHW format, [0,1] range)
    """
    os.makedirs(outdir, exist_ok=True)
    
    xyz = to_numpy(xyz_tensor)
    mask = to_numpy(mask_tensor)
    
    # Handle mask dimension
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    
    B = min(xyz.shape[0], max_samples)
    tb_images = {}
    
    for b in range(B):
        cur_xyz = xyz[b]  # [H, W, 3]
        cur_mask = (mask[b] > 0).astype(np.float32)
        
        # Extract channels
        x_chan = cur_xyz[:, :, 0]
        y_chan = cur_xyz[:, :, 1]
        z_chan = cur_xyz[:, :, 2]
        
        if apply_mask:
            # Normalize each channel
            x_norm = _normalize_for_visual(x_chan, mask=cur_mask, clip_percentile=(2, 98))
            y_norm = _normalize_for_visual(y_chan, mask=cur_mask, clip_percentile=(2, 98))
            z_norm = _normalize_for_visual(z_chan, mask=cur_mask, clip_percentile=(1, 99))
            
            # Apply mask (set invalid regions to 0)
            x_norm = x_norm * cur_mask
            y_norm = y_norm * cur_mask
            z_norm = z_norm * cur_mask
        else:
            # Normalize without mask
            x_norm = _normalize_for_visual(x_chan, mask=None, clip_percentile=(2, 98))
            y_norm = _normalize_for_visual(y_chan, mask=None, clip_percentile=(2, 98))
            z_norm = _normalize_for_visual(z_chan, mask=None, clip_percentile=(1, 99))
        
        # Disk filename: save as combined figure with 3 subplots
        if save_to_disk:
            disk_name = f"{prefix}_sample{b}_{suffix}_xyz"
            
            # Create figure with 3 subplots (1 row, 3 columns)
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # X channel
            im0 = axes[0].imshow(x_norm, cmap='seismic')
            axes[0].set_title(f'{suffix.upper()} X', fontsize=14)
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            # Y channel
            im1 = axes[1].imshow(y_norm, cmap='seismic')
            axes[1].set_title(f'{suffix.upper()} Y', fontsize=14)
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Z channel
            im2 = axes[2].imshow(z_norm, cmap='viridis')
            axes[2].set_title(f'{suffix.upper()} Z', fontsize=14)
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(f"{outdir}/{disk_name}.png", dpi=100, bbox_inches='tight')
            plt.close(fig)
        
        # Prepare images for TensorBoard (convert to CHW format)
        if b == 0:  # Only save first sample to TensorBoard to save space
            tb_images[f'{prefix}_xyz_{suffix}_x'] = torch.from_numpy(x_norm).unsqueeze(0)  # [1, H, W]
            tb_images[f'{prefix}_xyz_{suffix}_y'] = torch.from_numpy(y_norm).unsqueeze(0)
            tb_images[f'{prefix}_xyz_{suffix}_z'] = torch.from_numpy(z_norm).unsqueeze(0)

    # Additionally, create interactive 3D plotly visualization for first sample
    try:
        pc_sample = xyz[0]  # [H, W, 3]
        pc_vis = pc_sample.copy()
        if apply_mask:
            if mask is not None and mask.shape[0] > 0:
                cur_mask = mask[0]
                cur_mask_bool = (cur_mask > 0)
                if cur_mask_bool.shape == pc_vis.shape[:2]:
                    pc_vis[~cur_mask_bool] = np.nan
 
        pc_out = os.path.join(outdir, f"{prefix}_sample0_{suffix}_pc.html")
        out_dirname = os.path.dirname(pc_out)
        if out_dirname and not os.path.exists(out_dirname):
            os.makedirs(out_dirname, exist_ok=True)

        visualize_pointcloud_plotly(pc_vis, out_html=pc_out, max_points=200000, auto_open=False, colormap='Viridis')
    except Exception as e:
        print(f"Could not create interactive pointcloud: {e}")
    
    return tb_images


def visualize_depth_map(depth_tensor, outdir, prefix, idx, name='depth', max_samples=3, cmap='viridis', save_to_disk=True):
    """
    Visualize depth maps (z_p, z_c, etc.).
    
    Args:
        depth_tensor: [B, H, W, 1] or [B, H, W] tensor
        outdir: output directory
        prefix: filename prefix
        idx: iteration index
        name: name for this depth map (e.g., 'z_p', 'z_c')
        max_samples: max samples to save
        cmap: colormap
        save_to_disk: whether to save to disk (filename without iter)
    
    Returns:
        Dictionary of images for TensorBoard
    """
    os.makedirs(outdir, exist_ok=True)
    
    depth = to_numpy(depth_tensor)
    
    # Handle dimension
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    
    B = min(depth.shape[0], max_samples)
    tb_images = {}
    
    for b in range(B):
        cur_depth = depth[b]
        
        # Normalize
        depth_norm = _normalize_for_visual(cur_depth, mask=None, clip_percentile=(1, 99))
        
        # Disk filename: no iter number (will overwrite)
        if save_to_disk:
            disk_name = f"{prefix}_sample{b}_{name}"
            save_single_channel_image(depth_norm, f"{outdir}/{disk_name}.png", cmap=cmap)
        
        # TensorBoard (only first sample)
        if b == 0:
            tb_images[f'{prefix}_{name}'] = torch.from_numpy(depth_norm).unsqueeze(0)
    
    return tb_images


def visualize_image(img_tensor, outdir, prefix, idx, name='image', max_samples=3, is_grayscale=True, save_to_disk=True):
    """
    Visualize images (Ic_scaled, Ip_coded, etc.).
    
    Args:
        img_tensor: [B, H, W, C] or [B, H, W] tensor
        outdir: output directory
        prefix: filename prefix
        idx: iteration index
        name: name for this image
        max_samples: max samples to save
        is_grayscale: whether to treat as grayscale
        save_to_disk: whether to save to disk (filename without iter)
    
    Returns:
        Dictionary of images for TensorBoard
    """
    os.makedirs(outdir, exist_ok=True)
    
    img = to_numpy(img_tensor)
    
    # Handle dimension
    if img.ndim == 4 and img.shape[-1] == 1:
        img = img[..., 0]
        is_grayscale = True
    
    B = min(img.shape[0], max_samples)
    tb_images = {}
    
    for b in range(B):
        cur_img = img[b]
        
        # Clip to [0, 1] if not already
        cur_img = np.clip(cur_img, 0.0, 1.0)
        
        # Disk filename: no iter number (will overwrite)
        if save_to_disk:
            disk_name = f"{prefix}_sample{b}_{name}"
            
            if is_grayscale:
                save_single_channel_image(cur_img, f"{outdir}/{disk_name}.png", cmap='gray')
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(cur_img)
                ax.axis('off')
                plt.tight_layout()
                plt.savefig(f"{outdir}/{disk_name}.png", dpi=100, bbox_inches='tight')
                plt.close(fig)
        
        # TensorBoard (only first sample)
        if b == 0:
            if is_grayscale:
                tb_images[f'{prefix}_{name}'] = torch.from_numpy(cur_img).unsqueeze(0)
            else:
                tb_images[f'{prefix}_{name}'] = torch.from_numpy(cur_img).permute(2, 0, 1)
    
    return tb_images


def visualize_all(xyz_pred, z_p, z_c, Ic_scaled, Ip_coded, mask, 
                  xyz_gt=None, z_p_crop=None, z_c_crop=None, Ic_scaled_crop=None,
                  outdir='./recon', prefix='train', idx=0, max_samples=3, save_to_disk=True, apply_mask_vis=True):
    """
    Comprehensive visualization of all key variables.
    按要求组织：
    - GT: x, y, z 在同一个图里（各自有标题）
    - Inputs: Ic_scaled_crop, Ip_coded, mask 在同一个图里（各自有标题）
    - Pred: x, y, z 在同一个图里（各自有标题）
    - Depths: z_c_crop, z_p_crop 在同一个图里（各自有标题）
    - 移除：Ic_scaled (full), z_p (full), z_c (full), pred_rgb, gt_rgb
    
    Args:
        xyz_pred: [B, H, W, 3] predicted xyz coordinates
        z_p: [B, H, W, 1] projector depth map (full size)
        z_c: [B, H, W, 1] camera depth map (full size)
        Ic_scaled: [B, H, W, 1] scaled camera image (full size)
        Ip_coded: [B, H, W, 1] coded projector pattern
        mask: [B, H, W, 1] visibility mask
        xyz_gt: [B, H, W, 3] ground truth xyz coordinates (optional)
        z_p_crop: [B, H, W, 1] projector depth map (cropped, optional)
        z_c_crop: [B, H, W, 1] camera depth map (cropped, optional)
        Ic_scaled_crop: [B, H, W, 1] scaled camera image (cropped, optional)
        outdir: output directory
        prefix: 'train' or 'valid'
        idx: iteration index
        max_samples: max samples to save
        save_to_disk: whether to save to disk (filename without iter)
    
    Returns:
        Dictionary of all images for TensorBoard
    """
    os.makedirs(outdir, exist_ok=True)
    tb_images = {}
    
    # 1. Combined GT and Pred xyz: 2 rows x 3 columns (GT row + Pred row, each with x, y, z)
    if xyz_gt is not None and save_to_disk:
        xyz_np = to_numpy(xyz_pred)
        xyz_gt_np = to_numpy(xyz_gt)
        mask_np = to_numpy(mask)
        
        # Handle mask dimension
        if mask_np.ndim == 4 and mask_np.shape[-1] == 1:
            mask_np = mask_np[..., 0]
        
        B = min(xyz_np.shape[0], max_samples)
        
        for b in range(B):
            cur_xyz = xyz_np[b]  # [H, W, 3]
            cur_xyz_gt = xyz_gt_np[b]
            cur_mask = (mask_np[b] > 0).astype(np.float32)
            
            # Extract and normalize GT channels
            gt_x = _normalize_for_visual(cur_xyz_gt[:, :, 0], mask=cur_mask, clip_percentile=(2, 98)) * cur_mask
            gt_y = _normalize_for_visual(cur_xyz_gt[:, :, 1], mask=cur_mask, clip_percentile=(2, 98)) * cur_mask
            gt_z = _normalize_for_visual(cur_xyz_gt[:, :, 2], mask=cur_mask, clip_percentile=(1, 99)) * cur_mask
            
            # Extract and normalize Pred channels
            pred_x = _normalize_for_visual(cur_xyz[:, :, 0], mask=cur_mask, clip_percentile=(2, 98)) * cur_mask
            pred_y = _normalize_for_visual(cur_xyz[:, :, 1], mask=cur_mask, clip_percentile=(2, 98)) * cur_mask
            pred_z = _normalize_for_visual(cur_xyz[:, :, 2], mask=cur_mask, clip_percentile=(1, 99)) * cur_mask
            
            # Create figure with 2 rows x 3 columns
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            # GT row
            im00 = axes[0, 0].imshow(gt_x, cmap='seismic')
            axes[0, 0].set_title('GT X', fontsize=14)
            axes[0, 0].axis('off')
            plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)
            
            im01 = axes[0, 1].imshow(gt_y, cmap='seismic')
            axes[0, 1].set_title('GT Y', fontsize=14)
            axes[0, 1].axis('off')
            plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)
            
            im02 = axes[0, 2].imshow(gt_z, cmap='viridis')
            axes[0, 2].set_title('GT Z', fontsize=14)
            axes[0, 2].axis('off')
            plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04)
            
            # Pred row
            im10 = axes[1, 0].imshow(pred_x, cmap='seismic')
            axes[1, 0].set_title('Pred X', fontsize=14)
            axes[1, 0].axis('off')
            plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            im11 = axes[1, 1].imshow(pred_y, cmap='seismic')
            axes[1, 1].set_title('Pred Y', fontsize=14)
            axes[1, 1].axis('off')
            plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            im12 = axes[1, 2].imshow(pred_z, cmap='viridis')
            axes[1, 2].set_title('Pred Z', fontsize=14)
            axes[1, 2].axis('off')
            plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(f"{outdir}/{prefix}_sample{b}_xyz_comparison.png", dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # For TensorBoard (only first sample)
            if b == 0:
                tb_images[f'{prefix}_xyz_gt_x'] = torch.from_numpy(gt_x).unsqueeze(0)
                tb_images[f'{prefix}_xyz_gt_y'] = torch.from_numpy(gt_y).unsqueeze(0)
                tb_images[f'{prefix}_xyz_gt_z'] = torch.from_numpy(gt_z).unsqueeze(0)
                tb_images[f'{prefix}_xyz_pred_x'] = torch.from_numpy(pred_x).unsqueeze(0)
                tb_images[f'{prefix}_xyz_pred_y'] = torch.from_numpy(pred_y).unsqueeze(0)
                tb_images[f'{prefix}_xyz_pred_z'] = torch.from_numpy(pred_z).unsqueeze(0)
        
        # Interactive 3D point clouds for first sample
        try:
            # GT point cloud
            pc_gt = xyz_gt_np[0].copy()
            cur_mask_bool = (mask_np[0] > 0)
            pc_gt[~cur_mask_bool] = np.nan
            pc_gt_out = os.path.join(outdir, f"{prefix}_sample0_gt_pc.html")
            visualize_pointcloud_plotly(pc_gt, out_html=pc_gt_out, max_points=200000, auto_open=False, colormap='Viridis')
            
            # Pred point cloud
            pc_pred = xyz_np[0].copy()
            pc_pred[~cur_mask_bool] = np.nan
            pc_pred_out = os.path.join(outdir, f"{prefix}_sample0_pred_pc.html")
            visualize_pointcloud_plotly(pc_pred, out_html=pc_pred_out, max_points=200000, auto_open=False, colormap='Viridis')
        except Exception as e:
            print(f"Could not create interactive pointcloud: {e}")
    
    elif save_to_disk:  # Only pred, no GT
        xyz_pred_imgs = visualize_xyz_prediction(xyz_pred, mask, outdir, prefix, idx, max_samples, save_to_disk, suffix='pred', apply_mask=apply_mask_vis)
        tb_images.update(xyz_pred_imgs)
    
    # 3. Inputs: Ic_scaled_crop, Ip_coded, mask in one figure (3 subplots)
    if save_to_disk:
        B = min(xyz_pred.shape[0], max_samples) if hasattr(xyz_pred, 'shape') else max_samples
        for b in range(B):
            # Prepare data
            Ic_crop_np = to_numpy(Ic_scaled_crop[b:b+1] if Ic_scaled_crop is not None else Ic_scaled[b:b+1])
            Ip_np = to_numpy(Ip_coded[b:b+1])
            mask_np = to_numpy(mask[b:b+1])
            
            # Handle dimensions
            if Ic_crop_np.ndim == 4 and Ic_crop_np.shape[-1] == 1:
                Ic_crop_np = Ic_crop_np[0, :, :, 0]
            elif Ic_crop_np.ndim == 4:
                Ic_crop_np = Ic_crop_np[0]
            elif Ic_crop_np.ndim == 3:
                Ic_crop_np = Ic_crop_np[0] if Ic_crop_np.shape[0] == 1 else Ic_crop_np
            
            if Ip_np.ndim == 4 and Ip_np.shape[-1] == 1:
                Ip_np = Ip_np[0, :, :, 0]
            elif Ip_np.ndim == 4:
                Ip_np = Ip_np[0]
            elif Ip_np.ndim == 3:
                Ip_np = Ip_np[0] if Ip_np.shape[0] == 1 else Ip_np
                
            if mask_np.ndim == 4 and mask_np.shape[-1] == 1:
                mask_np = mask_np[0, :, :, 0]
            elif mask_np.ndim == 4:
                mask_np = mask_np[0]
            elif mask_np.ndim == 3:
                mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np
            
            # Clip to [0,1]
            Ic_crop_np = np.clip(Ic_crop_np, 0, 1)
            Ip_np = np.clip(Ip_np, 0, 1)
            mask_np = np.clip(mask_np, 0, 1)
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Ic_scaled_crop
            axes[0].imshow(Ic_crop_np, cmap='gray')
            axes[0].set_title('Ic_scaled_crop', fontsize=14)
            axes[0].axis('off')
            
            # Ip_coded
            axes[1].imshow(Ip_np, cmap='gray')
            axes[1].set_title('Ip_coded', fontsize=14)
            axes[1].axis('off')
            
            # Mask
            axes[2].imshow(mask_np, cmap='gray')
            axes[2].set_title('Mask', fontsize=14)
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{outdir}/{prefix}_sample{b}_inputs.png", dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # For TensorBoard (only first sample)
            if b == 0:
                tb_images[f'{prefix}_Ic_scaled_crop'] = torch.from_numpy(Ic_crop_np).unsqueeze(0)
                tb_images[f'{prefix}_Ip_coded'] = torch.from_numpy(Ip_np).unsqueeze(0)
                tb_images[f'{prefix}_mask'] = torch.from_numpy(mask_np).unsqueeze(0)
    
    # 4. Cropped depths: z_c_crop, z_p_crop in one figure (2 subplots)
    if z_c_crop is not None and z_p_crop is not None and save_to_disk:
        B = min(z_c_crop.shape[0], max_samples)
        for b in range(B):
            zc_np = to_numpy(z_c_crop[b:b+1])
            zp_np = to_numpy(z_p_crop[b:b+1])
            
            # Handle dimensions
            if zc_np.ndim == 4 and zc_np.shape[-1] == 1:
                zc_np = zc_np[0, :, :, 0]
            elif zc_np.ndim == 4:
                zc_np = zc_np[0]
            elif zc_np.ndim == 3:
                zc_np = zc_np[0] if zc_np.shape[0] == 1 else zc_np
                
            if zp_np.ndim == 4 and zp_np.shape[-1] == 1:
                zp_np = zp_np[0, :, :, 0]
            elif zp_np.ndim == 4:
                zp_np = zp_np[0]
            elif zp_np.ndim == 3:
                zp_np = zp_np[0] if zp_np.shape[0] == 1 else zp_np
            
            # Normalize
            zc_norm = _normalize_for_visual(zc_np, mask=None, clip_percentile=(1, 99))
            zp_norm = _normalize_for_visual(zp_np, mask=None, clip_percentile=(1, 99))
            
            # Create figure with 2 subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # z_c_crop
            im0 = axes[0].imshow(zc_norm, cmap='viridis')
            axes[0].set_title('z_c_crop', fontsize=14)
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            # z_p_crop
            im1 = axes[1].imshow(zp_norm, cmap='viridis')
            axes[1].set_title('z_p_crop', fontsize=14)
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(f"{outdir}/{prefix}_sample{b}_depths.png", dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # For TensorBoard (only first sample)
            if b == 0:
                tb_images[f'{prefix}_z_c_crop'] = torch.from_numpy(zc_norm).unsqueeze(0)
                tb_images[f'{prefix}_z_p_crop'] = torch.from_numpy(zp_norm).unsqueeze(0)
    
    return tb_images

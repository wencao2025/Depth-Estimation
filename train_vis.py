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


def visualize_xyz_prediction(xyz_tensor, mask_tensor, outdir, prefix, idx, max_samples=3, save_to_disk=True):
    """
    Visualize xyz prediction with separate x, y, z channels and RGB composite.
    
    Args:
        xyz_tensor: [B, H, W, 3] tensor (NHWC format)
        mask_tensor: [B, H, W, 1] or [B, H, W] mask tensor
        outdir: output directory
        prefix: filename prefix (e.g., 'train', 'valid')
        idx: iteration/step index
        max_samples: maximum number of samples to save from batch
        save_to_disk: whether to save to disk (filename without iter)
    
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
        
        # Normalize each channel
        x_norm = _normalize_for_visual(x_chan, mask=cur_mask, clip_percentile=(2, 98))
        y_norm = _normalize_for_visual(y_chan, mask=cur_mask, clip_percentile=(2, 98))
        z_norm = _normalize_for_visual(z_chan, mask=cur_mask, clip_percentile=(1, 99))
        
        # Apply mask (set invalid regions to 0)
        x_norm = x_norm * cur_mask
        y_norm = y_norm * cur_mask
        z_norm = z_norm * cur_mask
        
        # Disk filename: no iter number (will overwrite)
        if save_to_disk:
            disk_name = f"{prefix}_sample{b}"
            save_single_channel_image(x_norm, f"{outdir}/{disk_name}_x.png", cmap='viridis')
            save_single_channel_image(y_norm, f"{outdir}/{disk_name}_y.png", cmap='viridis')
            save_single_channel_image(z_norm, f"{outdir}/{disk_name}_z.png", cmap='viridis')
            
            # Create RGB composite (x->R, y->G, z->B)
            rgb = np.stack([x_norm, y_norm, z_norm], axis=-1)
            rgb_masked = rgb * np.expand_dims(cur_mask, -1)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(rgb_masked)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"{outdir}/{disk_name}_rgb.png", dpi=100, bbox_inches='tight')
            plt.close(fig)
        
        # Prepare images for TensorBoard (convert to CHW format)
        if b == 0:  # Only save first sample to TensorBoard to save space
            tb_images[f'{prefix}_xyz_x'] = torch.from_numpy(x_norm).unsqueeze(0)  # [1, H, W]
            tb_images[f'{prefix}_xyz_y'] = torch.from_numpy(y_norm).unsqueeze(0)
            tb_images[f'{prefix}_xyz_z'] = torch.from_numpy(z_norm).unsqueeze(0)
            tb_images[f'{prefix}_xyz_rgb'] = torch.from_numpy(rgb_masked).permute(2, 0, 1)  # [3, H, W]
    
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
                  z_p_crop=None, z_c_crop=None, Ic_scaled_crop=None,
                  outdir='./recon', prefix='train', idx=0, max_samples=3, save_to_disk=True):
    """
    Comprehensive visualization of all key variables.
    
    Args:
        xyz_pred: [B, H, W, 3] predicted xyz coordinates
        z_p: [B, H, W, 1] projector depth map (full size)
        z_c: [B, H, W, 1] camera depth map (full size)
        Ic_scaled: [B, H, W, 1] scaled camera image (full size)
        Ip_coded: [B, H, W, 1] coded projector pattern
        mask: [B, H, W, 1] visibility mask
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
    tb_images = {}
    
    # 1. Visualize xyz prediction (x, y, z separately + RGB)
    xyz_imgs = visualize_xyz_prediction(xyz_pred, mask, outdir, prefix, idx, max_samples, save_to_disk)
    tb_images.update(xyz_imgs)
    
    # 2. Visualize z_p (projector depth - full size)
    zp_imgs = visualize_depth_map(z_p, outdir, prefix, idx, name='z_p', max_samples=1, cmap='viridis', save_to_disk=save_to_disk)
    tb_images.update(zp_imgs)
    
    # 3. Visualize z_c (camera depth - full size)
    zc_imgs = visualize_depth_map(z_c, outdir, prefix, idx, name='z_c', max_samples=1, cmap='viridis', save_to_disk=save_to_disk)
    tb_images.update(zc_imgs)
    
    # 4. Visualize z_p_crop (projector depth - cropped)
    if z_p_crop is not None:
        zp_crop_imgs = visualize_depth_map(z_p_crop, outdir, prefix, idx, name='z_p_crop', max_samples=max_samples, cmap='viridis', save_to_disk=save_to_disk)
        tb_images.update(zp_crop_imgs)
    
    # 5. Visualize z_c_crop (camera depth - cropped)
    if z_c_crop is not None:
        zc_crop_imgs = visualize_depth_map(z_c_crop, outdir, prefix, idx, name='z_c_crop', max_samples=max_samples, cmap='viridis', save_to_disk=save_to_disk)
        tb_images.update(zc_crop_imgs)
    
    # 6. Visualize Ic_scaled (camera image)
    ic_imgs = visualize_image(Ic_scaled, outdir, prefix, idx, name='Ic_scaled', max_samples=1, is_grayscale=True, save_to_disk=save_to_disk)
    tb_images.update(ic_imgs)
    
    # 7. Visualize Ip_coded (projector pattern)
    ip_imgs = visualize_image(Ip_coded, outdir, prefix, idx, name='Ip_coded', max_samples=1, is_grayscale=True, save_to_disk=save_to_disk)
    tb_images.update(ip_imgs)
    
    # 8. Visualize mask
    mask_imgs = visualize_image(mask, outdir, prefix, idx, name='mask', max_samples=max_samples, is_grayscale=True, save_to_disk=save_to_disk)
    tb_images.update(mask_imgs)
    
    # 9. Visualize Ic_scaled_crop (cropped camera image)
    if Ic_scaled_crop is not None:
        ic_crop_imgs = visualize_image(Ic_scaled_crop, outdir, prefix, idx, name='Ic_scaled_crop', max_samples=max_samples, is_grayscale=True, save_to_disk=save_to_disk)
        tb_images.update(ic_crop_imgs)
    
    return tb_images

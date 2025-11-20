
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
import torch

def save_three_panel_gif(xyz, out_path='xyz_three_panel.gif', normalize='per_channel', fps=2.0, duration=None):
    """
    xyz: torch.Tensor or numpy array with shape [T,H,W,3]
    Handles tensors that require grad by detaching first.
    """
    # convert torch -> numpy safely
    if isinstance(xyz, torch.Tensor):
        # detach if needed, move to cpu, convert to numpy
        arr = xyz.detach().cpu().numpy()
    else:
        arr = np.array(xyz)

    if arr.ndim != 4 or arr.shape[3] != 3:
        raise ValueError(f"Expected shape [T,H,W,3], got {arr.shape}")

    T, H, W, C = arr.shape

    # compute normalization ranges
    if normalize == 'per_channel':
        mins = arr.min(axis=(0,1,2))
        maxs = arr.max(axis=(0,1,2))
    elif normalize == 'global':
        mins = arr.min()
        maxs = arr.max()
    else:
        raise ValueError("normalize must be 'per_channel' or 'global'")

    # choose labeling font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    frames = []
    for t in range(T):
        panels = []
        for c in range(C):
            img = arr[t, :, :, c].astype(np.float32)
            if normalize == 'per_channel':
                vmin = float(mins[c]); vmax = float(maxs[c])
            else:
                vmin = float(mins); vmax = float(maxs)
            if vmax > vmin:
                norm = (img - vmin) / (vmax - vmin)
            else:
                norm = np.zeros_like(img, dtype=np.float32)
            img8 = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)
            pil = Image.fromarray(img8, mode='L').convert('RGB')
            panels.append(pil)
        combined = Image.new('RGB', (W * C, H))
        for i, p in enumerate(panels):
            combined.paste(p, (i * W, 0))

        # add frame label like '1/10' at top-left
        draw = ImageDraw.Draw(combined)
        text = f"{t+1}/{T}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad = 6
        draw.rectangle([(5, 5), (5 + text_w + pad, 5 + text_h + pad)], fill='black')
        draw.text((8, 5), text, fill='white', font=font)

        frames.append(np.array(combined))

    # write GIF: prefer fps if provided, otherwise fall back to duration per-frame
    if fps is not None:
        imageio.mimsave(out_path, frames, fps=float(fps), loop=0)
    else:
        # duration is seconds per frame
        imageio.mimsave(out_path, frames, duration=duration, loop=0)
    print(f"Saved three-panel GIF with {T} frames to: {out_path}")

try:
    # example: save at 2 FPS and include per-frame labels
    save_three_panel_gif(xyz_cView_crop_hat, out_path='./vis_debug_pattern/xyz_three_panel.gif', normalize='per_channel', fps=2.0)
except NameError:
    print(" `xyz_cView_crop_hat` not found in the current scope. Please ensure you run this code in the Debug Console at the current breakpoint/scope, or change the variable name to the one you are actually using.")
except Exception as e:
    print("Runtime error:", repr(e))
    raise


############################################################################
import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import imageio

save_dir = './vis_debug_pattern'
os.makedirs(save_dir, exist_ok=True)

def tensor_to_numpy(tensor):
    """Convert tensor to numpy array, handling various cases"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)

def normalize_for_vis(data):
    """Normalize to [0, 255] uint8"""
    data = data.astype(np.float32)
    data_min, data_max = data.min(), data.max()
    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min) * 255
    else:
        data = np.zeros_like(data)
    return data.astype(np.uint8)

def save_png(data, filepath, name):
    """Save a single PNG image"""
    data_np = tensor_to_numpy(data)
    
    # Take the first batch (if any)
    while data_np.ndim > 3:
        data_np = data_np[0]
    
    # Remove single channel dimension
    if data_np.ndim == 3 and data_np.shape[-1] == 1:
        data_np = data_np[..., 0]
    elif data_np.ndim == 3 and data_np.shape[0] == 1:
        data_np = data_np[0]
    
    data_norm = normalize_for_vis(data_np)
    img = Image.fromarray(data_norm)
    img.save(filepath)
    print(f"✓ Saved {name}: {filepath}")

def save_gif_with_labels(data, filepath, name, label_prefix="Frame"):
    """Save a GIF animation with labels, using the last dimension as frames (depth as frames)"""
    data_np = tensor_to_numpy(data)
    
    # Take the first batch (if the first dimension is batch)
    if data_np.ndim == 5:  # [B, H, W, C, D] or [B, H, W, D, C]
        data_np = data_np[0]
    elif data_np.ndim == 4:  # [B, H, W, D] or [H, W, C, D]
        # If the first dimension looks like batch size (usually < 10)
        if data_np.shape[0] <= 10 and data_np.shape[0] < data_np.shape[-1]:
            data_np = data_np[0]  # remove batch
    
    # Now should be [H, W, D] or [H, W, C, D] format
    # The last dimension is the number of depth layers
    num_frames = data_np.shape[-1]
    
    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    frames = []
    for i in range(num_frames):
        frame_data = data_np[..., i]  # [H, W] or [H, W, C]
        
        # Remove single channel dimension
        if frame_data.ndim == 3 and frame_data.shape[-1] == 1:
            frame_data = frame_data[..., 0]
        
        frame_norm = normalize_for_vis(frame_data)
        img = Image.fromarray(frame_norm).convert('RGB')
        
        # Add text label
        draw = ImageDraw.Draw(img)
        text = f"{label_prefix} {i+1}/{num_frames}"
        
        # Draw text background (black rectangle)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.rectangle([(5, 5), (15 + text_width, 15 + text_height)], fill='black')
        
        draw.text((10, 10), text, fill='white', font=font)
        
        frames.append(np.array(img))
    
    imageio.mimsave(filepath, frames, fps=2, loop=0)
    print(f"✓ Saved {name}: {filepath} ({num_frames} frames)")


def save_gif_firstdim(data, filepath, name, label_prefix="Frame"):
    """Save a GIF animation with labels, using the first dimension as frames (batch as frames)"""
    data_np = tensor_to_numpy(data)

    # If data has no batch dimension, use the last dimension as frames (fallback)
    if data_np.ndim == 3:
        # [H, W, D] -> use last dim as frames
        num_frames = data_np.shape[-1]
        use_last = True
    else:
        # Use first dim as frames
        num_frames = data_np.shape[0]
        use_last = False

    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    frames = []
    for i in range(num_frames):
        if use_last:
            frame_data = data_np[..., i]
        else:
            frame_data = data_np[i]

        # If [..., C] and C==1, remove channel
        if frame_data.ndim == 3 and frame_data.shape[-1] == 1:
            frame_data = frame_data[..., 0]

        # If [1, H, W] or [1, H, W, C]
        if frame_data.ndim == 4 and frame_data.shape[0] == 1:
            frame_data = frame_data[0]

        frame_norm = normalize_for_vis(frame_data)
        img = Image.fromarray(frame_norm).convert('RGB')

        # Add text label
        draw = ImageDraw.Draw(img)
        text = f"{label_prefix} {i+1}/{num_frames}"

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.rectangle([(5, 5), (15 + text_width, 15 + text_height)], fill='black')
        draw.text((10, 10), text, fill='white', font=font)

        frames.append(np.array(img))

    imageio.mimsave(filepath, frames, fps=2, loop=0)
    print(f"✓ Saved {name}: {filepath} ({num_frames} frames)")

# ===== Saving PNG images =====
print("\n=== Saving PNG images ===")

# 1. Ic
if 'Ic' in locals() or 'Ic' in globals():
    save_png(Ic, f'{save_dir}/Ic.png', 'Ic')

# 2. Ic_scaled
if 'Ic_scaled' in locals() or 'Ic_scaled' in globals():
    save_png(Ic_scaled, f'{save_dir}/Ic_scaled.png', 'Ic_scaled')

# 3. visible_mask_c_dense
if 'visible_mask_c_dense' in locals() or 'visible_mask_c_dense' in globals():
    save_png(visible_mask_c_dense, f'{save_dir}/visible_mask_c_dense.png', 'visible_mask_c_dense')

# 4. z_c
if 'z_c' in locals() or 'z_c' in globals():
    save_png(z_c, f'{save_dir}/z_c.png', 'z_c')

# 5. z_p
if 'z_p' in locals() or 'z_p' in globals():
    save_png(z_p, f'{save_dir}/z_p.png', 'z_p')

# 6. Ip_coded (saving as PNG)
if 'Ip_coded' in locals() or 'Ip_coded' in globals():
    save_png(Ip_coded, f'{save_dir}/Ip_coded.png', 'Ip_coded')

if 'Ic_mask' in locals() or 'Ic_mask' in globals():
    save_png(Ic_mask, f'{save_dir}/Ic_mask.png', 'Ic_mask')

# ===== Saving GIF animations =====
print("\n=== Saving GIF animations ===")

# 7. Ip (using last dimension)
if 'Ip' in locals() or 'Ip' in globals():
    save_gif_with_labels(Ip, f'{save_dir}/Ip.gif', 'Ip', label_prefix="Depth")

# 8. Ip_ref (using last dimension)
if 'Ip_ref' in locals() or 'Ip_ref' in globals():
    save_gif_with_labels(Ip_ref, f'{save_dir}/Ip_ref.gif', 'Ip_ref', label_prefix="Depth")

# ===== Saving batch-as-frames GIF (using first dimension) =====
if 'Ic_scaled_crop' in locals() or 'Ic_scaled_crop' in globals():
    save_gif_firstdim(Ic_scaled_crop, f'{save_dir}/Ic_scaled_crop.gif', 'Ic_scaled_crop', label_prefix="Frame")

if 'Ic_mask_crop' in locals() or 'Ic_mask_crop' in globals():
    save_gif_firstdim(Ic_mask_crop, f'{save_dir}/Ic_mask_crop.gif', 'Ic_mask_crop', label_prefix="Frame")

if 'z_c_crop' in locals() or 'z_c_crop' in globals():
    save_gif_firstdim(z_c_crop, f'{save_dir}/z_c_crop.gif', 'z_c_crop', label_prefix="Frame")

if 'Ic_crop' in locals() or 'Ic_crop' in globals():
    save_gif_firstdim(Ic_crop, f'{save_dir}/Ic_crop.gif', 'Ic_crop', label_prefix="Frame")

print("\n=== All visualizations saved! ===")


############################################################################
#########################save psfs vis code below##########################
############################################################################
import os, numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
import torch

# Variable names (change here if your variable names differ)
candidates = ['psfs', 'PSFs', 'psf', 'PSF']
psf_var = None
for n in candidates:
    if n in globals() or n in locals():
        psf_var = globals().get(n, locals().get(n))
        break
if psf_var is None:
    raise RuntimeError("Cannot find PSFs variable. Please define a variable named psfs/PSFs/psf/PSF in the Debug Console scope, or modify the candidate names in this snippet.")

# Output directory
out_dir = './vis_debug_pattern/psfs'
os.makedirs(out_dir, exist_ok=True)
gif_path = os.path.join(out_dir, 'psfs_firstdim.gif')

# Convert to numpy (supports torch.Tensor on GPU)
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

arr = to_numpy(psf_var)

# Supported shapes:
# [N, H, W], [N, H, W, 1], [N, H, W, C], also supports leading batch [1, N, H, W], etc. -> will try to find the first dimension as frames
if arr.ndim >= 4 and arr.shape[0] == 1 and arr.ndim == 4:
    # For example [1, N, H, W] -> becomes [N, H, W]
    arr = arr[0]

# If [H,W,D] (last dimension is depth), we want to save by the first dimension (batch), this case is not handled here — expecting first dimension as frames or N,H,W
if arr.ndim == 3:
    # Considered as [N, H, W] or [H, W, N] cannot be determined: we prefer [N,H,W] (first dimension is frames)
    n_frames = arr.shape[0]
    has_chan = False
elif arr.ndim == 4:
    # [N, H, W, C] or [N, C, H, W] possible, prefer [N, H, W, C]
    # If you think it's [N, C, H, W], please transpose in Debug Console first
    n_frames = arr.shape[0]
    has_chan = True
else:
    raise RuntimeError(f"Unsupported PSF array shape: {arr.shape}. Expected [N,H,W] or [N,H,W,C].")
# Whether to unify normalization across frames (True: use global min/max for all frames; False: normalize each frame independently)
unify_norm = True

if unify_norm:
    g_min = float(arr.min())
    g_max = float(arr.max())
else:
    g_min = g_max = None

# Font
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except:
    font = ImageFont.load_default()

frames = []
for i in range(n_frames):
    frame = arr[i]
    # If there is a channel dim and C==1, reduce to grayscale
    if has_chan and frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[..., 0]
    # Normalize to uint8
    f = frame.astype(np.float32)
    if g_min is None or g_max is None:
        mi, ma = f.min(), f.max()
    else:
        mi, ma = g_min, g_max
    if ma > mi:
        f_u8 = ((f - mi) / (ma - mi) * 255.0).astype(np.uint8)
    else:
        f_u8 = np.zeros_like(f, dtype=np.uint8)
    img = Image.fromarray(f_u8).convert('RGB')
    # Add label
    # draw = ImageDraw.Draw(img)
    # text = f"PSF {i+1}/{n_frames}"
    # bbox = draw.textbbox((0,0), text, font=font)
    # tw = bbox[2]-bbox[0]; th = bbox[3]-bbox[1]
    # draw.rectangle([(5,5),(10+tw,10+th)], fill='black')
    # draw.text((8,7), text, fill='white', font=font)
    frames.append(np.array(img))
    # Save individual PNG (optional)
    png_path = os.path.join(out_dir, f'psf_{i+1:03d}.png')
    Image.fromarray(f_u8).save(png_path)

# Save GIF (fps can be changed)
imageio.mimsave(gif_path, frames, fps=2, loop=0)
print(f"Saved PSF GIF -> {gif_path} and {n_frames} PNGs in {out_dir}")


############################################################################
##########################save pattern vis code below##########################
############################################################################
# Prerequisite: Ensure the project's import path is accessible (usually run debugging in the project root directory).
# Description: This snippet will try to import gen_pattern from the current scope or common modules,
#              then generate patterns for each patternMode and save the last dimension as frames in a GIF.

import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
import torch

save_base = './vis_debug_pattern/patterns'
os.makedirs(save_base, exist_ok=True)

# 1) Find gen_pattern function (prefer current scope, otherwise try importing common modules)
if 'gen_pattern' in globals() or 'gen_pattern' in locals():
    gen_pattern_fn = gen_pattern
else:
    # Try importing from training/tool modules
    tried = []
    gen_pattern_fn = None
    for mod in ('train_pytorch', 'utils_pytorch', 'train', 'utils'):
        try:
            m = __import__(mod)
            if hasattr(m, 'gen_pattern'):
                gen_pattern_fn = getattr(m, 'gen_pattern')
                break
        except Exception as e:
            tried.append((mod, str(e)))
    if gen_pattern_fn is None:
        raise RuntimeError("gen_pattern not found in globals and failed to import from common modules. Tried: " + ", ".join([t[0] for t in tried]))

# 2) helper: save tensor/ndarray as GIF along the last dimension (each frame with label)
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def normalize_frame(frame, global_min=None, global_max=None):
    f = frame.astype(np.float32)
    if global_min is None or global_max is None:
        mi, ma = f.min(), f.max()
    else:
        mi, ma = float(global_min), float(global_max)
    if ma > mi:
        f = (f - mi) / (ma - mi) * 255.0
    else:
        f = np.zeros_like(f)
    return f.astype(np.uint8)

def save_lastdim_gif(arr, out_path, label_prefix="Depth", fps=2, unify_norm=False):
    """
    arr: ndarray or torch tensor with shape [B?, H, W, D] or [H, W, D] or [H, W, C, D]
    Save gif where frames are along the last dim (D).
    """
    a = to_numpy(arr)
    # drop batch dim if present and batch==1
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    # now expect [H,W,D] or [H,W,C,D]
    if a.ndim == 3:
        H, W, D = a.shape
        has_chan = False
    elif a.ndim == 4:
        H, W, C, D = a.shape
        has_chan = True
    else:
        raise RuntimeError(f"Unexpected array shape {a.shape} for saving last-dim gif.")
    # optionally compute global min/max for consistent normalization across frames
    gmin = gmax = None
    if unify_norm:
        if has_chan:
            # collapse channel when computing global min/max
            gmin = float(a.min())
            gmax = float(a.max())
        else:
            gmin = float(a.min())
            gmax = float(a.max())

    # load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    frames = []
    for i in range(a.shape[-1]):
        if has_chan:
            frame = a[..., i]  # H,W,C
            if frame.ndim == 3 and frame.shape[2] == 1:
                frame = frame[..., 0]
        else:
            frame = a[..., i]  # H,W
        frame_u8 = normalize_frame(frame, global_min=gmin, global_max=gmax)
        img = Image.fromarray(frame_u8).convert('RGB')
        draw = ImageDraw.Draw(img)
        text = f"{label_prefix} {i+1}/{a.shape[-1]}"
        bbox = draw.textbbox((0,0), text, font=font)
        tw = bbox[2]-bbox[0]; th = bbox[3]-bbox[1]
        draw.rectangle([(5,5),(10+tw,10+th)], fill='black')
        draw.text((8,7), text, fill='white', font=font)
        frames.append(np.array(img))
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"Saved GIF: {out_path} ({len(frames)} frames)")

# 3) parameters for gen_pattern - adjust if you want different values
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# B = 1
# H = 800
# W = 1200
# N_layers = 21
# grid_Isigma = 0.5
# stride = 16

pattern_modes = ['grid', 'kinect', 'MArray', 'kronTwoFix']

# 4) generate and save per mode
for mode in pattern_modes:
    try:
        print(f"Generating pattern for mode: {mode} ...")
        # ensure device arg is respected by gen_pattern
        # some gen_pattern implementations accept device as string or torch.device
        try:
            grid = gen_pattern_fn(B, H, W, N_layers, grid_Isigma, mode, stride, device=device)
        except TypeError:
            # fallback if gen_pattern signature uses device str
            grid = gen_pattern_fn(B, H, W, N_layers, grid_Isigma, mode, stride, device=str(device))
        # grid shape expected [B,H,W,D] or [1,H,W,D]
        out_file = os.path.join(save_base, f'pattern_{mode}.gif')
        save_lastdim_gif(grid, out_file, label_prefix=f"{mode}", fps=2, unify_norm=False)
    except Exception as e:
        print(f"Failed to generate/save for mode={mode}: {e}")



"""
Utilities to save `xyz_cView` (from Debug Console) to disk as `.npy` and `.ply`.

Usage in Debug Console:

    # Option A: call convenience function which auto-detects variable
    exec(open('vis_save_xyz_debug.py').read())
    save_xyz_from_debug_console(outdir='./vis_debug_pattern', base_name='xyz_cView_debug')

    # Option B: import and pass the tensor/array directly
    from vis_save_xyz_debug import save_xyz_npy, save_xyz_ply
    save_xyz_npy(xyz_cView, './vis_debug_pattern/xyz_cView.npy')
    save_xyz_ply(xyz_cView, './vis_debug_pattern/xyz_cView.ply')

The PLY writer writes ASCII PLY with X Y Z floats. If `xyz_cView` is a torch tensor it will be detached and moved to CPU automatically.
"""

import os
import sys
import numpy as np
import torch


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _to_ndarray(xyz):
    """Convert tensor or array to Nx3 numpy array of floats."""
    if torch.is_tensor(xyz):
        arr = xyz.detach().cpu().numpy()
    else:
        arr = np.array(xyz)

    # Handle batch dimension or H,W,3 shapes
    if arr.ndim == 4:  # [B, H, W, 3]
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[2] >= 3:  # [H, W, C]
        pts = arr[:, :, :3].reshape(-1, 3)
    elif arr.ndim == 2 and arr.shape[1] == 3:  # already Nx3
        pts = arr
    else:
        raise ValueError(f"Unsupported xyz shape: {arr.shape}")

    # Remove non-finite points
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]

    return pts.astype(np.float32)


def save_xyz_npy(xyz, out_path):
    """Save xyz to a .npy file. Returns path."""
    pts = _to_ndarray(xyz)
    _ensure_dir(out_path)
    np.save(out_path, pts)
    print(f"Saved .npy with {pts.shape[0]} points to: {out_path}")
    return out_path


def save_xyz_ply(xyz, out_path, include_color=False, color_map=None):
    """Save xyz to an ASCII PLY file. Returns path.

    include_color: if True, colors will be added per-vertex using color_map(z) or default gray.
    color_map: a callable mapping normalized z [0,1] -> (r,g,b) each in 0-255.
    """
    pts = _to_ndarray(xyz)
    _ensure_dir(out_path)
    n = pts.shape[0]

    # Optional color
    colors = None
    if include_color:
        z = pts[:, 2]
        vmin, vmax = np.percentile(z, 5), np.percentile(z, 95)
        zn = np.clip((z - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
        if color_map is None:
            # simple gray map
            colors = (np.stack([zn, zn, zn], axis=1) * 255).astype(np.uint8)
        else:
            colors = np.array([color_map(v) for v in zn], dtype=np.uint8)

    with open(out_path, 'w') as f:
        # PLY header
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {n}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if colors is not None:
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
        f.write('end_header\n')

        # write vertices
        if colors is None:
            for i in range(n):
                f.write(f"{pts[i,0]} {pts[i,1]} {pts[i,2]}\n")
        else:
            for i in range(n):
                r,g,b = colors[i]
                f.write(f"{pts[i,0]} {pts[i,1]} {pts[i,2]} {r} {g} {b}\n")

    print(f"Saved .ply with {n} points to: {out_path}")
    return out_path


def save_xyz_from_debug_console(outdir='./vis_debug_pattern', base_name='xyz_cView_debug',
                                save_npy=True, save_ply=True, ply_color=True, max_points_warn=500000):
    """Detect `xyz_cView` or `xyz_cView_crop_hat` in caller locals and save to disk.

    Returns: dict with saved paths.
    """
    frame = sys._getframe(1)
    local_vars = frame.f_locals

    candidates = ['xyz_cView', 'xyz_cView_crop_hat', 'xyz_cView_crop', 'xyz_cView_hat']
    found = None
    for k in candidates:
        if k in local_vars:
            found = (k, local_vars[k])
            break

    if found is None:
        print('Could not find xyz variable in current scope. Available vars:')
        print(list(local_vars.keys())[:50])
        return None

    name, xyz = found
    print(f"Found variable '{name}' in current scope. Saving...")

    # Ensure outdir
    os.makedirs(outdir, exist_ok=True)

    pts = _to_ndarray(xyz)
    n = pts.shape[0]
    if n == 0:
        print('No valid points to save.')
        return None
    if n > max_points_warn:
        print(f'Warning: point count {n} > {max_points_warn}. Consider subsampling before saving.')

    results = {}
    if save_npy:
        p_npy = os.path.join(outdir, base_name + '.npy')
        np.save(p_npy, pts)
        results['npy'] = p_npy
        print(f'Saved .npy: {p_npy} ({n} points)')

    if save_ply:
        p_ply = os.path.join(outdir, base_name + '.ply')
        save_xyz_ply(pts, p_ply, include_color=ply_color)
        results['ply'] = p_ply

    return results

'Utilities to save xyz_cView from debug console:'
"Example in debug console:"
" exec(open('vis_save_xyz_debug.py').read())"
" save_xyz_from_debug_console(outdir='./vis_debug_pattern', base_name='xyz_cView_debug')"

######################save coord_p and coord_c vis code below##########################
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def coord_to_points(coord, batch_idx=0, valid_z_thresh=1e-6):
    # coord: torch.Tensor [B,4,H,W] or numpy equivalent
    if torch.is_tensor(coord):
        arr = coord[batch_idx].permute(1,2,0).cpu().numpy()  # [H,W,4]
    else:
        arr = np.array(coord[batch_idx]).transpose(1,2,0)
    xyz = arr[..., :3].astype(np.float32)
    valid = np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > valid_z_thresh)
    pts = xyz[valid]
    return pts, valid, xyz

def save_ply(pts, filename='debug_pc.ply', max_points=200000):
    pts = np.asarray(pts)
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    with open(filename, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(pts)}\n')
        f.write('property float x\nproperty float y\nproperty float z\nend_header\n')
        for p in pts:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
    print('Saved PLY ->', filename)

def save_pointcloud_plotly(pts, out_html='debug_pc.html', max_points=200000, colormap='Viridis'):
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot
    except Exception as e:
        print('Plotly not available:', e)
        return
    pts = np.asarray(pts)
    if pts.shape[0] == 0:
        print('No points to save')
        return
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    x,y,z = pts[:,0], pts[:,1], pts[:,2]
    vmin, vmax = np.percentile(z, [2,98])
    znorm = np.clip((z - vmin) / (vmax - vmin + 1e-12), 0, 1)
    scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers',
                           marker=dict(size=1, color=znorm, colorscale=colormap, opacity=0.8))
    fig = go.Figure(data=[scatter])
    plot(fig, filename=out_html, auto_open=False)
    print('Saved interactive HTML ->', out_html)

def show_channel_maps(coord, batch_idx=0, clip_percent=(2,98), fname=''):
    if torch.is_tensor(coord):
        arr = coord[batch_idx].permute(1,2,0).cpu().numpy()
    else:
        arr = np.array(coord[batch_idx]).transpose(1,2,0)
    x = arr[...,0]; y = arr[...,1]; z = arr[...,2]
    def norm(i):
        v = i[np.isfinite(i)]
        if v.size == 0:
            return np.zeros_like(i)
        lo, hi = np.percentile(v, clip_percent)
        out = (i - lo) / (hi - lo + 1e-12)
        return np.clip(out, 0, 1)
    xn, yn, zn = norm(x), norm(y), norm(z)
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    ax[0].imshow(xn, cmap='seismic'); ax[0].set_title('X'); ax[0].axis('off')
    ax[1].imshow(yn, cmap='seismic'); ax[1].set_title('Y'); ax[1].axis('off')
    ax[2].imshow(zn, cmap='viridis'); ax[2].set_title('Z'); ax[2].axis('off')
    plt.show()
    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved channel maps -> {fname}")

def scatter_projections(pts, sample=200000):
    pts = np.asarray(pts)
    if pts.shape[0] == 0:
        print('No points to plot'); return
    if pts.shape[0] > sample:
        idx = np.random.choice(pts.shape[0], sample, replace=False)
        pts = pts[idx]
    fig, axes = plt.subplots(1,3,figsize=(18,5))
    sc = axes[0].scatter(pts[:,0], pts[:,1], c=pts[:,2], s=0.5, cmap='viridis')
    axes[0].set_title('XY (top)'); axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
    plt.colorbar(sc, ax=axes[0], fraction=0.046)
    sc2 = axes[1].scatter(pts[:,1], pts[:,2], c=pts[:,0], s=0.5, cmap='viridis')
    axes[1].set_title('YZ (side)'); axes[1].set_xlabel('Y'); axes[1].set_ylabel('Z')
    plt.colorbar(sc2, ax=axes[1], fraction=0.046)
    sc3 = axes[2].scatter(pts[:,0], pts[:,2], c=pts[:,1], s=0.5, cmap='viridis')
    axes[2].set_title('XZ (front)'); axes[2].set_xlabel('X'); axes[2].set_ylabel('Z')
    for ax in axes: ax.set_aspect('equal', 'box')
    plt.tight_layout(); plt.show()
    os.makedirs('./vis_debug_pattern', exist_ok=True)
    out_path = os.path.join('./vis_debug_pattern', f'scatter_projections_sample{min(sample, pts.shape[0])}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved scatter projections ->', out_path)

# ---- Example usage in debug console ----
pts_p, valid_p, _ = coord_to_points(coord_p, batch_idx=0)
show_channel_maps(coord_p, batch_idx=0, fname='./vis_debug_pattern/coord_p_channel_maps.png')
scatter_projections(pts_p, sample=150000)
# save_ply(pts_p, 'coord_p_sample.ply')
save_pointcloud_plotly(pts_p, './vis_debug_pattern/coord_p_plot.html')

pst_c, valid_c, _ = coord_to_points(coord_c, batch_idx=0)
show_channel_maps(coord_c, batch_idx=0, fname='./vis_debug_pattern/coord_c_channel_maps.png')
scatter_projections(pst_c, sample=150000)
# save_ply(pst_c, 'coord_c_sample.ply')
save_pointcloud_plotly(pst_c, './vis_debug_pattern/coord_c_plot.html')

#######################save illum gaussian#####################
import numpy as np
from PIL import Image
import torch
import os


arr = base_pattern.detach().cpu().numpy()

if arr.ndim == 3 and arr.shape[2] == 1:
    arr = arr[..., 0]
if arr.ndim == 3 and arr.shape[0] == 1:
    arr = arr[0]


valid = np.isfinite(arr)
if valid.any():
    lo, hi = np.percentile(arr[valid], [0, 100])
else:
    lo, hi = arr.min(), arr.max()
norm = (arr - lo) / (hi - lo + 1e-12)
norm = np.clip(norm, 0.0, 1.0)

img_uint8 = (norm * 255.0).astype(np.uint8)
im = Image.fromarray(img_uint8, mode='L')   # 'L' 表示灰度
im.save('./vis_debug_pattern/base_pattern_gray_2400x3600.png')
print('Saved ./vis_debug_pattern/base_pattern_gray_2400x3600.png')
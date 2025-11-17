'''
Function for optics related, mainly for pattern blur
PyTorch version
'''

import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import os

#### optical parameters:
wvl = 530e-9

# for Wm -60to60
z_min = 0.695714566
z_max = 0.941062289
z0 = 0.8
D = 21 / 2 * 1.4 * 1e-3

# These are the experimental calibrated PSFs.
fn_psfs = 'psfs_exp.npy'
if os.path.exists(fn_psfs):
    PSFs_np = np.load(fn_psfs)
else:
    print(f"Warning: {fn_psfs} not found. Using dummy PSFs.")
    PSFs_np = np.ones((21, 5, 5, 1), dtype=np.float32)  # Dummy PSFs


def gen_WmMask(WmMap, Wm_layers, device='cuda'):
    """
    Generate binary mask for different Wm layers
    Args:
        WmMap: Wavefront error map [B, H, W, 1]
        Wm_layers: Array of Wm layer values
        device: torch device
    Returns:
        mask: Binary mask [B, H, W, N_layers]
    """
    N_layers = len(Wm_layers)
    Wm_step = (Wm_layers[-1] - Wm_layers[0]) / (N_layers - 1)  # find the step size of Wm_layers
    
    # Convert to integer step size
    Wm_layers_normalized = Wm_layers / Wm_step
    WmMap_normalized = WmMap / Wm_step
    
    B, H, W, _ = WmMap.shape
    mask_list = []
    
    for Wm in Wm_layers_normalized:
        mask0 = (torch.abs(torch.round(WmMap_normalized) - Wm) < 0.01).float()
        mask_list.append(mask0)
    
    mask = torch.cat(mask_list, dim=3)  # [B, H, W, N_layers]
    return mask


def z2Wm(z):
    """
    Convert depth z to wavefront error Wm
    Args:
        z: depth map (can be numpy array or torch tensor)
    Returns:
        Wm: wavefront error map (same type as input)
    """
    k = 2 * np.pi / wvl
    Wm = k * (D / 2) ** 2 / 2 * (1 / z - 1 / z0)
    return Wm


def codePattern(pattern, zMap, device='cuda', is_conv_psf=True):
    """
    Apply pattern blur based on depth map using PSFs
    Args:
        pattern: Input pattern [B, H, W, N_layers]
        zMap: Depth map [B, H, W, 1]
        device: torch device
    Returns:
        Ip_coded_norm: Coded pattern with blur [B, H, W, 1]
    """
    N_layers = pattern.shape[-1]
    Wm_min = z2Wm(z_max)
    Wm_max = z2Wm(z_min)
    Wm_layers = np.linspace(Wm_min, Wm_max, N_layers)
    
    WmMap = z2Wm(zMap)
    WmMask = gen_WmMask(WmMap, Wm_layers, device)  # binary mask to see which layer each point locates
    
    # Load PSFs
    PSFs = torch.from_numpy(PSFs_np).float().to(device)  # [N_layers, H_psf, W_psf, 1]
    
    # PyTorch conv2d expects [B, C_in, H, W] for input and [C_out, C_in, kH, kW] for weight
    # TF: pattern[:, :, :, 0:1] is [B, H, W, 1]
    # TF: PSFs after transpose [1, 2, 3, 0] becomes [H_psf, W_psf, 1, N_layers]
    
    # Convert pattern to PyTorch format
    pattern_single = pattern[:, :, :, 0:1]  # [B, H, W, 1]
    pattern_t = pattern_single.permute(0, 3, 1, 2)  # [B, 1, H, W]
    
    if is_conv_psf:
        # Convert PSFs to PyTorch conv2d weight format
        # Original PSFs: [N_layers, H_psf, W_psf, 1]
        # After TF transpose [1, 2, 3, 0]: [H_psf, W_psf, 1, N_layers]
        # PyTorch needs: [C_out=N_layers, C_in=1, kH, kW]
        PSFs_t = PSFs.permute(0, 3, 1, 2)  # [N_layers, 1, H_psf, W_psf]
        
        # Apply convolution (this outputs N_layers channels)
        conv_out = F.conv2d(pattern_t, PSFs_t, padding='same')  # [B, N_layers, H, W]
        conv_out = conv_out.permute(0, 2, 3, 1)  # [B, H, W, N_layers]
    else:
        # If not convolving with PSF, just replicate pattern across layers
        conv_out = pattern[:, :, :, 0:1]  # [B, H, W, 1]
    
    # Multiply by mask and sum across layers
    Ip_coded = torch.sum(conv_out * WmMask, dim=-1, keepdim=True)  # [B, H, W, 1]
    
    # Normalize to max = 1
    norm = torch.max(torch.max(Ip_coded, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0]
    Ip_coded_norm = Ip_coded / (norm + 1e-10)
    
    return Ip_coded_norm


def codePattern_simple(pattern, zMap):
    """
    Simplified version without PSF convolution (for testing/fast computation)
    Args:
        pattern: Input pattern [B, H, W, 1] or [B, H, W, N_layers]
        zMap: Depth map [B, H, W, 1]
    Returns:
        pattern: Simply returns the input pattern (no blur applied)
    """
    # Simplified version: just return the pattern without PSF blur
    if pattern.shape[-1] > 1:
        # If multi-layer, return first layer
        return pattern[:, :, :, 0:1]
    return pattern


# 
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # z2Wm
    z_test = torch.tensor([[[[0.7]], [[0.8]], [[0.9]]]], device=device)
    Wm_test = z2Wm(z_test)
    print(f"z2Wm test:")
    print(f"  z: {z_test.squeeze().cpu().numpy()}")
    print(f"  Wm: {Wm_test.squeeze().cpu().numpy()}")
    
    # gen_WmMask
    B, H, W = 1, 100, 100
    WmMap = torch.randn(B, H, W, 1, device=device) * 60  # Wm range approximately -60 to 60
    Wm_layers = np.linspace(-60, 60, 21)
    mask = gen_WmMask(WmMap, Wm_layers, device)
    print(f"\ngen_WmMask test:")
    print(f"  WmMap shape: {WmMap.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask sum per pixel (should be ~1): {mask.sum(dim=-1).mean().item():.4f}")
    
    # codePattern
    pattern = torch.rand(B, H, W, 21, device=device)
    zMap = torch.rand(B, H, W, 1, device=device) * (z_max - z_min) + z_min
    
    if os.path.exists(fn_psfs):
        Ip_coded = codePattern(pattern, zMap, device)
        print(f"\ncodePattern test:")
        print(f"  Pattern shape: {pattern.shape}")
        print(f"  zMap shape: {zMap.shape}")
        print(f"  Coded pattern shape: {Ip_coded.shape}")
        print(f"  Coded pattern range: [{Ip_coded.min().item():.4f}, {Ip_coded.max().item():.4f}]")
    else:
        print(f"\nSkipping codePattern test (PSFs file not found)")
    
    # codePattern_simple
    pattern_simple = torch.rand(B, H, W, 1, device=device)
    Ip_simple = codePattern_simple(pattern_simple, zMap)
    print(f"\ncodePattern_simple test:")
    print(f"  Input shape: {pattern_simple.shape}")
    print(f"  Output shape: {Ip_simple.shape}")
    print(f"  Output is same as input: {torch.allclose(pattern_simple, Ip_simple)}")
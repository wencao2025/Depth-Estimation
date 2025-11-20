import os
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio


##########################################   generate grid pattern on the projector  #############################################

def gen_pattern(B, H, W, N_layers, grid_Isigma, patternMode, stride, device='cuda', dot_grid_size=None, radius=None):
    """
    Generate projector pattern
    Args:
        B: batch size
        H, W: height and width
        N_layers: number of pattern layers
        grid_Isigma: standard deviation for grid intensity
        patternMode: pattern type ('grid', 'kinect', 'MArray', 'kronTwoFix', 'dotArray')
        stride: stride for pattern generation
        device: torch device
        dot_grid_size: tuple (n_rows, n_cols) for dotArray mode, e.g., (50, 75) means 50×75 dots
    Returns:
        grid: [B, H, W, N_layers] pattern tensor
    """
    grid_intensity = F.relu(torch.randn(B, 1, 1, 1, device=device) * grid_Isigma + 1)
    
    if patternMode == "grid":
        grid0 = np.zeros([B, H + 2 * stride, W + 2 * stride, N_layers], np.float32)
        grid0[:, 0: -1: stride, :, :] = 1
        grid0[:, :, 0: -1: stride, :] = 1
        grid0_tensor = torch.from_numpy(grid0).to(device)
        # Random crop
        h_start = torch.randint(0, 2 * stride + 1, (1,)).item()
        w_start = torch.randint(0, 2 * stride + 1, (1,)).item()
        grid1 = grid0_tensor[:, h_start:h_start+H, w_start:w_start+W, :]
        
    elif patternMode == "kinect":
        kinect_mat = sio.loadmat('ProjPatterns/kinect1200.mat')
        kinect_np = kinect_mat['binary']
        grid1 = torch.from_numpy(kinect_np).float().to(device).view(1, H, W, 1).repeat(B, 1, 1, N_layers)
        
    elif patternMode == "MArray":
        kinect_mat = sio.loadmat('ProjPatterns/M_array.mat')
        kinect_np = kinect_mat['squares']
        grid1 = torch.from_numpy(kinect_np).float().to(device).view(1, H, W, 1).repeat(B, 1, 1, N_layers)
        
    elif patternMode == "kronTwoFix":
        ## local pattern
        # cross
        Pla = np.zeros([stride, stride, 1, 1], np.float32)
        Pla[int(stride / 2), :, :, :] = 1
        Pla[:, int(stride / 2), :, :] = 1
        Pla = torch.from_numpy(Pla).to(device)
        
        # square
        Plb = np.zeros([stride, stride, 1, 1], np.float32)
        Plb[int(stride / 4), int(stride / 4):int(3 * stride / 4), :, :] = 1
        Plb[int(3 * stride / 4), int(stride / 4):int(3 * stride / 4), :, :] = 1
        Plb[int(stride / 4):int(3 * stride / 4), int(stride / 4), :, :] = 1
        Plb[int(stride / 4):int(3 * stride / 4) + 1, int(3 * stride / 4), :, :] = 1
        Plb = torch.from_numpy(Plb).to(device)
        
        ## global pattern
        if stride == 16:
            Pg = torch.from_numpy(np.load('ProjPatterns/kron_Pg_50x75_0.5.npy')).float().to(device)
        else:
            raise ValueError('the stride has to be 16 for kronTwoFix.')
        
        # PyTorch conv_transpose2d: input [N,C,H,W], weight [C_in, C_out, kH, kW]
        # TF conv2d_transpose: input [B,H,W,C], filter [kH,kW,C_out,C_in]
        Pg_t = Pg.permute(0, 3, 1, 2)  # [1, C, H, W]
        Pla_t = Pla.permute(3, 2, 0, 1)  # [C_out, C_in, kH, kW]
        Plb_t = Plb.permute(3, 2, 0, 1)
        
        grid0a = F.conv_transpose2d(Pg_t, Pla_t, stride=stride)
        grid0b = F.conv_transpose2d(1 - Pg_t, Plb_t, stride=stride)
        
        # Crop to exact size
        grid0a = grid0a[:, :, :H, :W]
        grid0b = grid0b[:, :, :H, :W]
        
        grid_combined = (grid0a + grid0b).permute(0, 2, 3, 1)  # [B, H, W, C]
        grid1 = grid_combined.repeat(B, 1, 1, N_layers)
        
    elif patternMode == "dotArray":
        # Generate dot array pattern with multiple layers
        # Use dot_grid_size if provided, otherwise fallback to stride-based spacing
        dot_grid_size=(100,150)
        if dot_grid_size is not None:
            n_rows, n_cols = dot_grid_size
            # Calculate spacing to evenly distribute dots across H × W
            spacing_h = H / n_rows
            spacing_w = W / n_cols
            dot_size = max(2, int(min(spacing_h, spacing_w) / 3))  # dot size is ~1/3 of spacing
        else:
            # Fallback to original stride-based method
            dot_spacing = stride * 2
            spacing_h = spacing_w = dot_spacing
            dot_size = max(2, stride // 3)
        
        base_pattern = torch.zeros(H, W, device=device)
        
        # Create dots at calculated positions
        if dot_grid_size is not None:
            # Evenly distributed grid
            for i in range(n_rows):
                for j in range(n_cols):
                    # Center position of each dot
                    i_center = int((i + 0.5) * spacing_h)
                    j_center = int((j + 0.5) * spacing_w)
                    
                    # Draw circular dot
                    for di in range(-dot_size // 2, dot_size // 2 + 1):
                        for dj in range(-dot_size // 2, dot_size // 2 + 1):
                            if di * di + dj * dj <= (dot_size // 2) ** 2:
                                i_pos = i_center + di
                                j_pos = j_center + dj
                                if 0 <= i_pos < H and 0 <= j_pos < W:
                                    base_pattern[i_pos, j_pos] = 1.0
        else:
            # Original stride-based regular intervals
            for i in range(0, H, int(spacing_h)):
                for j in range(0, W, int(spacing_w)):
                    i_center = min(i + dot_size // 2, H - 1)
                    j_center = min(j + dot_size // 2, W - 1)
                    
                    # Draw circular dot
                    for di in range(-dot_size // 2, dot_size // 2 + 1):
                        for dj in range(-dot_size // 2, dot_size // 2 + 1):
                            if di * di + dj * dj <= (dot_size // 2) ** 2:
                                i_pos = i_center + di
                                j_pos = j_center + dj
                                if 0 <= i_pos < H and 0 <= j_pos < W:
                                    base_pattern[i_pos, j_pos] = 1.0
        # Repeat for all batches and layers (no random variation)
        grid1 = base_pattern.unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1, N_layers)
    elif patternMode == "TrainedDotArray":
        data = np.load('dotarray_pattern.npz', allow_pickle=True)
        pattern_data = data['psf']    
        pattern_hwn = np.transpose(pattern_data, (1, 2, 0))  
        pattern_bhwn = np.expand_dims(pattern_hwn, 0)    # shape (1, H, W, N)
        pattern_bhwn = np.repeat(pattern_bhwn, B, axis=0)  # shape (B, H, W, N) # shape (H, W, N)        # shape [N_layers, H, W]
        grid1 = torch.from_numpy(pattern_bhwn).float().to(device='cuda' if torch.cuda.is_available() else 'cpu')
        data.close()
    elif patternMode == "GaussianDot":
        # Generate dot array pattern with multiple layers.
        # Use `dot_grid_size` (n_rows, n_cols) to control number of dots and
        # `radius` (in pixels) to control each Gaussian dot's radius.
        if dot_grid_size is None:
            n_rows, n_cols = 120, 180
        else:
            n_rows, n_cols = dot_grid_size

        # Calculate spacing to evenly distribute dots across H × W
        spacing_h = float(H) / float(n_rows)
        spacing_w = float(W) / float(n_cols)

        # determine radius in pixels
        if radius is None:
            # default radius roughly proportional to spacing
            radius_px = max(1, int(min(spacing_h, spacing_w) / 6))
        else:
            radius_px = int(max(1, round(radius)))

        # choose gaussian sigma (controls blur). Use sigma = radius/2 by default
        sigma = max(0.5, float(radius_px) / 2.0)

        # kernel half-size
        r = int(radius_px)
        ky = 2 * r + 1
        kx = 2 * r + 1

        # create Gaussian kernel (isotropic)
        yv = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
        xv = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(yv, xv, indexing='ij')
        kernel = torch.exp(-((xx ** 2) / (2 * sigma ** 2) + (yy ** 2) / (2 * sigma ** 2)))
        kernel = kernel / kernel.max()

        base_pattern = torch.zeros(H, W, device=device, dtype=torch.float32)

        # place Gaussian dots on a regular grid centered in each cell
        for i in range(n_rows):
            for j in range(n_cols):
                i_center = int((i + 0.5) * spacing_h)
                j_center = int((j + 0.5) * spacing_w)

                # kernel window bounds in image coordinates
                i0 = max(0, i_center - r)
                i1 = min(H, i_center + r + 1)
                j0 = max(0, j_center - r)
                j1 = min(W, j_center + r + 1)

                # corresponding kernel slice
                ki0 = i0 - (i_center - r)
                ki1 = ki0 + (i1 - i0)
                kj0 = j0 - (j_center - r)
                kj1 = kj0 + (j1 - j0)

                base_pattern[i0:i1, j0:j1] += kernel[ki0:ki1, kj0:kj1]

        

        # Repeat for all batches and layers (no random variation)
        grid1 = base_pattern.unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1, N_layers)

    else:
        raise ValueError(f"Unknown patternMode: {patternMode}")
    
    grid = grid1 * grid_intensity
    return grid


#######################################  loss function  #############################################

def cost_rms_mask(GT, hat, mask, eps=1e-6):
    """
    RMS loss with mask
    Args:
        GT: ground truth [B, H, W, C]
        hat: prediction [B, H, W, C]
        mask: binary mask [B, H, W, C]
        eps: small value to prevent numerical instability
    Returns:
        loss: scalar
    """
    # loss = torch.sqrt(torch.sum(torch.square(mask * (GT - hat))) / (torch.sum(mask) + 1))
    # Add eps to prevent sqrt(0) gradient issues and clamp denominator
    numerator = torch.sum(torch.square(mask * (GT - hat)))
    denominator = torch.sum(mask).clamp(min=eps)
    loss = torch.sqrt(numerator / denominator + eps)
    return loss


def cost_L1_mask(GT, hat, mask):
    """
    L1 loss with mask
    Args:
        GT: ground truth [B, H, W, C]
        hat: prediction [B, H, W, C]
        mask: binary mask [B, H, W, C]
    Returns:
        loss: scalar
    """
    loss = torch.sum(mask * torch.abs(GT - hat)) / (torch.sum(mask) + 1)
    return loss


def cost_grad_mask(GT, hat, mask):
    """
    Gradient loss with mask
    Args:
        GT: ground truth [B, H, W, C]
        hat: prediction [B, H, W, C]
        mask: binary mask [B, H, W, C]
    Returns:
        loss: scalar
    """
    # Expand mask using avg pool
    mask_t = mask.permute(0, 3, 1, 2)  # [B, C, H, W]
    mask_expand_t = F.avg_pool2d(mask_t, kernel_size=9, stride=1, padding=4)
    mask_expand = (mask_expand_t > 0.999).float().permute(0, 2, 3, 1)  # [B, H, W, C]
    
    # Compute gradients
    GTy, GTx = image_gradients(GT)
    haty, hatx = image_gradients(hat)
    
    costx = cost_rms_mask(GTx, hatx, mask_expand)
    costy = cost_rms_mask(GTy, haty, mask_expand)
    
    return costx + costy


def image_gradients(images):
    """
    Compute image gradients (dy, dx)
    Args:
        images: [B, H, W, C]
    Returns:
        dy: [B, H, W, C] vertical gradients
        dx: [B, H, W, C] horizontal gradients
    """
    dy = images[:, 1:, :, :] - images[:, :-1, :, :]
    dx = images[:, :, 1:, :] - images[:, :, :-1, :]
    
    # Pad to maintain shape
    dy = F.pad(dy, (0, 0, 0, 0, 0, 1), mode='constant', value=0)
    dx = F.pad(dx, (0, 0, 0, 1, 0, 0), mode='constant', value=0)
    
    return dy, dx


#####################################   crop image to multiple random patches  ################################

def multi_rand_crop(I, Ic_size, offset):
    """
    Random crop with given offsets
    Args:
        I: input image [B, H, W, C]
        Ic_size: crop size
        offset: [B_sub, 2] offset positions (h, w)
    Returns:
        Ic: [B_sub, Ic_size, Ic_size, C] cropped images
    """
    B, H, W, C = I.shape
    crops = []
    for i in range(offset.shape[0]):
        h_start = offset[i, 0].item()
        w_start = offset[i, 1].item()
        crop = I[0:1, h_start:h_start+Ic_size, w_start:w_start+Ic_size, :]
        crops.append(crop)
    
    Ic = torch.cat(crops, dim=0)
    return Ic


#####################################   adjust the pattern intensity based on the reflectance  ################################

def depth2SN(depth, pixelsize=200e-6):
    """
    Get surface normal from depth
    Args:
        depth: depth map [B, H, W, C]
        pixelsize: pixel size in meters
    Returns:
        theta: surface normal angle
    """
    dy, dx = image_gradients(depth)
    dM = torch.sqrt(dy ** 2 + dx ** 2)
    tan = dM / pixelsize
    theta = torch.atan(tan)
    return theta


def withReflectance(I, depth):
    """
    Change the intensity of the pattern based on the reflectance
    Args:
        I: input pattern [B, H, W, C]
        depth: depth map [B, H, W, C]
    Returns:
        I_ref: pattern with reflectance adjustment
    """
    SN = depth2SN(depth, 100e-6)
    reflectance = torch.cos(SN) * 0.8 + 0.2  # add a DC and cosine term
    I_ref = I * reflectance
    
    return I_ref


##########################################   local normalization  #############################################

def ln(Im, kernelsize=17, constant=0.01):
    """
    Local normalization
    Args:
        Im: input image [B, H, W, C]
        kernelsize: kernel size for average pooling
        constant: small constant to avoid division by zero
    Returns:
        norm: normalized image
    """
    # Convert to [B, C, H, W] for avg_pool2d
    Im_t = Im.permute(0, 3, 1, 2)
    padding = kernelsize // 2
    mean = F.avg_pool2d(Im_t, kernel_size=kernelsize, stride=1, padding=padding)
    mean = mean.permute(0, 2, 3, 1)  # Back to [B, H, W, C]
    norm = Im / (mean + constant)
    return norm


##########################################   add random texture from the scene  #############################################

def addTexture(Ip_coded, variation):
    """
    Add random texture to the coded pattern
    Args:
        Ip_coded: coded pattern [B, H, W, C]
        variation: variation range
    Returns:
        textured pattern
    """
    B, H, W, C = Ip_coded.shape
    device = Ip_coded.device
    
    # Create a small random pattern and resize it
    base_h = torch.randint(2, 20, (1,)).item()
    base_w = torch.randint(2, 20, (1,)).item()
    texture0 = torch.rand(base_h, base_w, device=device) * variation + (1 - variation)
    texture1 = texture0.unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1, C)  # [B, h, w, C]
    
    # Scale up using different interpolation
    texture1_t = texture1.permute(0, 3, 1, 2)  # [B, C, h, w]
    if torch.rand(1).item() > 0.5:
        texture2 = F.interpolate(texture1_t, size=[W, W], mode='bilinear', align_corners=False)
    else:
        texture2 = F.interpolate(texture1_t, size=[W, W], mode='nearest')
    
    # Random crop
    h_start = torch.randint(0, W - H + 1, (1,)).item() if W > H else 0
    w_start = torch.randint(0, W - W + 1, (1,)).item()
    texture3 = texture2[:, :, h_start:h_start+H, :W].permute(0, 2, 3, 1)  # [B, H, W, C]
    
    return Ip_coded * texture3


def addTexture_checkerboard(Ip_coded, contrast):
    """
    Add checkerboard texture to the coded pattern
    Args:
        Ip_coded: coded pattern [B, H, W, C]
        contrast: contrast ratio
    Returns:
        textured pattern
    """
    B, H, W, C = Ip_coded.shape
    device = Ip_coded.device
    
    # Create a small pattern and resize it
    size = 5
    texture0 = np.ones((B, size, size, C), dtype=np.float32) / contrast
    texture0[:, 1::2, ::2, :] = 1  # variation
    texture0[:, ::2, 1::2, :] = 1  # variation
    texture1 = torch.from_numpy(texture0).to(device)
    
    # Scale up
    texture1_t = texture1.permute(0, 3, 1, 2)  # [B, C, h, w]
    texture2 = F.interpolate(texture1_t, size=[W, W], mode='nearest')
    
    # Crop to exact size
    texture3 = texture2[:, :, :H, :W].permute(0, 2, 3, 1)  # [B, H, W, C]
    
    return Ip_coded * texture3


def upsample_depth_by_factor(z, scale_factor, mode='bilinear', align_corners=False):
    """
    Upsample spatial dims H,W of a depth tensor `z` shaped [B, H, W, 1].

    Args:
        z: torch.Tensor with shape [B, H, W, 1]
        scale_factor: float scale or tuple (H_new, W_new) target size or int factor
        mode: interpolation mode, default 'bilinear' (use 'nearest' for masks or discrete labels)
        align_corners: passed to F.interpolate for bilinear/bicubic modes

    Returns:
        z_up: torch.Tensor with shape [B, H_new, W_new, 1]
    """
    if not torch.is_tensor(z):
        z = torch.tensor(z)

    if z.ndim != 4 or z.shape[-1] != 1:
        raise ValueError("z must have shape [B, H, W, 1]")

    # Permute to [B, C=1, H, W]
    z_t = z.permute(0, 3, 1, 2).contiguous()
    _, _, H, W = z_t.shape

    # Determine target size
    if isinstance(scale_factor, (tuple, list)) and len(scale_factor) == 2:
        H_new, W_new = int(scale_factor[0]), int(scale_factor[1])
    else:
        # treat as numeric scale factor
        factor = float(scale_factor)
        H_new = int(round(H * factor))
        W_new = int(round(W * factor))

    # Interpolate
    if mode in ('bilinear', 'bicubic'):
        z_up_t = F.interpolate(z_t, size=(H_new, W_new), mode=mode, align_corners=align_corners)
    else:
        z_up_t = F.interpolate(z_t, size=(H_new, W_new), mode=mode)

    # Back to [B, H_new, W_new, 1]
    z_up = z_up_t.permute(0, 2, 3, 1).contiguous()
    return z_up


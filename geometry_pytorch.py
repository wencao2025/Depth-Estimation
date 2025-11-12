'''
Geometric functions. Parameters are for Blender simulation images with 800x1200 resolution
PyTorch version
'''

import numpy as np
import torch
import torch.nn.functional as F
import math

H = 800
W = 1200


def pixel_to_ray(pixel, pixel_width=1200, pixel_height=800):
    """
    Convert pixel coordinate to ray direction
    Args:
        pixel: (x, y) pixel coordinate
        pixel_width, pixel_height: image dimensions
    Returns:
        (x_vect, y_vect, 1.0) ray direction
    """
    x, y = pixel
    x_vect = pixel_width / (2*1666.67) * ((2.0 * ((x + 0.5) / pixel_width)) - 1.0)
    y_vect = pixel_height / (2*1666.67) * ((2.0 * ((y + 0.5) / pixel_height)) - 1.0)
    return (x_vect, y_vect, 1.0)


def pixel_to_ray_array(width=1200, height=800):
    """
    Create array of ray directions for all pixels
    For this un-normalized version, final ray z = 1
    Returns:
        array of shape [height, width, 3]
    """
    pixel_to_ray_array = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            pixel_to_ray_array[y, x] = np.array(pixel_to_ray((x, y), pixel_height=height, pixel_width=width))
    return pixel_to_ray_array


def points_in_camera_coords(z_map, pixel_to_ray_array, allBatch=False):
    """
    Convert z_map to 4D homogeneous coordinates
    Args:
        z_map: depth map [B, H, W, 1]
        pixel_to_ray_array: ray directions [H, W, 3] or [B, H, W, 3]
        allBatch: if True, pixel_to_ray_array already has batch dimension
    Returns:
        camera coordinates [B, H, W, 4]
    """
    B, H, W, _ = z_map.shape
    z_map3 = z_map.repeat(1, 1, 1, 3)  # [B, H, W, 3]
    
    if allBatch:
        pixel_to_ray_array_batch = pixel_to_ray_array.float()
    else:
        pixel_to_ray_array_batch = torch.from_numpy(pixel_to_ray_array).float().to(z_map.device)
        pixel_to_ray_array_batch = pixel_to_ray_array_batch.unsqueeze(0).repeat(B, 1, 1, 1)
    
    camera_relative_xyz = z_map3 * pixel_to_ray_array_batch
    ones = torch.ones(B, H, W, 1, device=z_map.device)
    return torch.cat([camera_relative_xyz, ones], dim=-1)


def coord_transform(coord, transform):
    """
    Transforms 4D coordinate to 2D pixel coordinate based on the transformation matrix.
    Args:
        coord: [B, 4, H, W]
        transform: [B, 4, 4]
    Returns:
        2D pixel coordinate, shape = [B, H, W, 2]
    """
    B, _, H, W = coord.shape
    coord = coord.reshape(B, 4, -1)  # [B, 4, H*W]
    unnormalized_pixel_coords = torch.matmul(transform, coord)  # [B, 4, H*W]
    
    x_u = unnormalized_pixel_coords[:, 0:1, :]  # [B, 1, H*W]
    y_u = unnormalized_pixel_coords[:, 1:2, :]
    z_u = unnormalized_pixel_coords[:, 2:3, :]
    
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    
    pixel_coords = torch.cat([x_n, y_n], dim=1)  # [B, 2, H*W]
    pixel_coords = pixel_coords.reshape(B, 2, H, W)
    return pixel_coords.permute(0, 2, 3, 1)  # [B, H, W, 2]


def coord0TOgrid1(coord0, pose01):
    """
    Project 4D space coordinate in view 0 to pixel coordinate in view 1.
    Args:
        coord0: 4D space coordinate in view 0, shape = [B, 4, H, W]
        pose01: relative pose transformation from view 0 to view 1, shape = [B, 4, 4]
    Returns:
        grid1: the 2D pixel coordinate in view 1, shape = [B, H, W, 2]
    """
    B = coord0.shape[0]
    intrinsics = get_intrinsic(B, device=coord0.device)
    transform = torch.matmul(intrinsics, pose01)
    grid1 = coord_transform(coord0, transform)
    return grid1


def bilinear_sampler(imgs, coords):
    """
    Construct a new image by bilinear sampling from the input image.
    Points falling outside the source image boundary have value 0.
    
    Args:
        imgs: source image to be sampled from [batch, height_s, width_s, channels]
        coords: coordinates of source pixels to sample from [batch, height_t, width_t, 2]
                The two channels correspond to x and y coordinates respectively.
    Returns:
        A new sampled image [batch, height_t, width_t, channels]
    """
    B, H_s, W_s, C = imgs.shape
    B, H_t, W_t, _ = coords.shape
    
    coords_x = coords[:, :, :, 0:1]  # [B, H_t, W_t, 1]
    coords_y = coords[:, :, :, 1:2]
    
    x0 = torch.floor(coords_x)
    x1 = x0 + 1
    y0 = torch.floor(coords_y)
    y1 = y0 + 1
    
    y_max = float(H_s - 1)
    x_max = float(W_s - 1)
    zero = torch.tensor(0.0, device=imgs.device)
    
    x0_safe = torch.clamp(x0, zero, x_max)
    y0_safe = torch.clamp(y0, zero, y_max)
    x1_safe = torch.clamp(x1, zero, x_max)
    y1_safe = torch.clamp(y1, zero, y_max)
    
    # Weights
    wt_x0 = x1_safe - coords_x
    wt_x1 = coords_x - x0_safe
    wt_y0 = y1_safe - coords_y
    wt_y1 = coords_y - y0_safe
    
    # Indices in the flat image
    dim2 = float(W_s)
    dim1 = float(W_s * H_s)
    
    # Base indices for each batch
    base = torch.arange(B, device=imgs.device).float() * dim1
    base = base.view(B, 1, 1, 1).repeat(1, H_t, W_t, 1)  # [B, H_t, W_t, 1]
    
    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    
    idx00 = (x0_safe + base_y0).reshape(-1).long()
    idx01 = (x0_safe + base_y1).reshape(-1).long()
    idx10 = (x1_safe + base_y0).reshape(-1).long()
    idx11 = (x1_safe + base_y1).reshape(-1).long()
    
    # Sample from imgs
    imgs_flat = imgs.reshape(-1, C)  # [B*H_s*W_s, C]
    
    im00 = imgs_flat[idx00].reshape(B, H_t, W_t, C)
    im01 = imgs_flat[idx01].reshape(B, H_t, W_t, C)
    im10 = imgs_flat[idx10].reshape(B, H_t, W_t, C)
    im11 = imgs_flat[idx11].reshape(B, H_t, W_t, C)
    
    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1
    
    output = w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11
    return output


def gen_visible_mask(grid, method='round'):
    """
    Creates a visible mask by (forward) warping.
    Variable `grid` provides many 2D locations, mark them (nearest grid) as visible.
    
    Args:
        grid: 2D pixel coordinate, shape = [B, H, W, 2]
        method: "all" or "round". "all" means to mark 1 as many as possible. 
                "round" means to mark 1 as few as possible.
    Returns:
        visible_mask: a 2D visible mask, shape = [B, H, W, 1]
    """
    B, H, W, _ = grid.shape
    device = grid.device
    
    if method == 'all':
        grid_idx_ff = torch.floor(grid[:, :, :, 0]) + torch.floor(grid[:, :, :, 1]) * W
        grid_idx_fc = torch.floor(grid[:, :, :, 0]) + torch.ceil(grid[:, :, :, 1]) * W
        grid_idx_cf = torch.ceil(grid[:, :, :, 0]) + torch.floor(grid[:, :, :, 1]) * W
        grid_idx_cc = torch.ceil(grid[:, :, :, 0]) + torch.ceil(grid[:, :, :, 1]) * W
        
        grid_idx = torch.stack([grid_idx_ff, grid_idx_fc, grid_idx_cf, grid_idx_cc], dim=3)  # [B, H, W, 4]
        grid_idx_1d = grid_idx.reshape(B, -1).long()
    elif method == 'round':
        grid_idx = torch.round(grid[:, :, :, 0]) + torch.round(grid[:, :, :, 1]) * W
        grid_idx_1d = grid_idx.reshape(B, -1).long()
    
    # Create mask for each batch
    mask_list = []
    for i in range(B):
        mask_i = torch.zeros(H * W, dtype=torch.bool, device=device)
        valid_idx = grid_idx_1d[i]
        # Filter out invalid indices
        valid_mask = (valid_idx >= 0) & (valid_idx < H * W)
        valid_idx = valid_idx[valid_mask]
        if len(valid_idx) > 0:
            mask_i[valid_idx] = True
        mask_list.append(mask_i.unsqueeze(0))
    
    mask = torch.cat(mask_list, dim=0)  # [B, H*W]
    visible_mask = mask.float().reshape(B, H, W, 1)
    
    return visible_mask


def z2pointcloud(z):
    """
    Converts zMap (the actual value in 3D point cloud) to 4D coordinate (x, y, z, 1)
    Args:
        z: z map in Cartesian coordinate, shape = [B, H, W, 1]
    Returns:
        coord: 4D coordinate, shape = [B, 4, H, W]
    """
    cached_pixel_to_ray_array = pixel_to_ray_array()
    coord = points_in_camera_coords(z, cached_pixel_to_ray_array).permute(0, 3, 1, 2)
    return coord


def get_intrinsic(B, device='cuda'):
    """
    Get camera intrinsic matrix for the sceneNet dataset
    Args:
        B: batch size
        device: torch device
    Returns:
        Intrinsic matrix [B, 4, 4]
    """
    Intrinsic = np.diag([1666.67, 1666.67, 1, 1])
    Intrinsic[0][2] = 1200/2
    Intrinsic[1][2] = 800/2
    Intrinsic = torch.from_numpy(Intrinsic).float().to(device)
    Intrinsic = Intrinsic.unsqueeze(0).repeat(B, 1, 1)  # [B, 4, 4]
    return Intrinsic


def warp_p2c(Ip, coord_p, coord_c, pose_p2c):
    """
    Warps the image Ip (projector view) to Ic (camera view).
    
    Args:
        Ip: Image from the projector view, shape = [B, H, W, 1]
        coord_p: 4D scene coordinate in projector view, shape = [B, 4, H, W]
        coord_c: 4D scene coordinate in camera view, shape = [B, 4, H, W]
        pose_p2c: camera pose transformation from projector to camera, shape = [B, 4, 4]
    
    Returns:
        Ic_masked: Warped image with occluded regions to be black, shape = [B, H, W, 1]
        grid_p2c: For coord_p, the corresponding 2D pixel coordinate in camera view, shape = [B, H, W, 2]
        grid_c2p: For coord_c, the corresponding 2D pixel coordinate in projector view, shape = [B, H, W, 2]
    """
    # Inverse warp to get intensity in camera view
    pose_c2p = torch.linalg.inv(pose_p2c)
    grid_c2p = coord0TOgrid1(coord_c, pose_c2p)
    Ic = bilinear_sampler(Ip, grid_c2p)  # inverse warp
    
    # Forward warp to get mask in camera view
    grid_p2c = coord0TOgrid1(coord_p, pose_p2c)
    visible_mask_c = gen_visible_mask(grid_p2c, 'all')
    
    Ic_masked = Ic * visible_mask_c
    
    return Ic_masked, grid_p2c, grid_c2p


def zp_cView_to_zc(zp_cView, pose_p2c, xy):
    """
    Convert zp in camera view to zc in camera view
    Args:
        zp_cView: depth from projector in camera view [B, H, W, 1]
        pose_p2c: pose transformation from projector to camera [B, 4, 4]
        xy: x/z and y/z coordinates [B, H, W, 2]
    Returns:
        zc: depth from camera [B, H, W, 1]
    """
    pose_c2p = torch.linalg.inv(pose_p2c)
    a20 = pose_c2p[:, 2:3, 0:1]  # [B, 1, 1]
    a21 = pose_c2p[:, 2:3, 1:2]
    a22 = pose_c2p[:, 2:3, 2:3]
    a23 = pose_c2p[:, 2:3, 3:4]
    
    # Reshape for broadcasting
    a20 = a20.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
    a21 = a21.view(-1, 1, 1, 1)
    a22 = a22.view(-1, 1, 1, 1)
    a23 = a23.view(-1, 1, 1, 1)
    
    zc = (zp_cView - a23) / (a20 * xy[:, :, :, 0:1] + a21 * xy[:, :, :, 1:2] + a22)
    
    return zc
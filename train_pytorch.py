'''
Main code to train the networks - PyTorch version
'''
'''
cd /mnt/ssd1/wencao/project/Depth_Estimation
git add - A
git commit -m "描述你的修改"
git push
查看状态：git status
查看修改：git diff
查看提交历史：git log --oneline
拉取更新：git pull
'''
# tmux list-sessions
# tmux attach-session -t myjob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from geometry_pytorch import (warp_p2c, z2pointcloud, bilinear_sampler, 
                               gen_visible_mask, coord0TOgrid1, pixel_to_ray_array,
                               points_in_camera_coords, zp_cView_to_zc)
from data_provider_pytorch import read_data
from network_pytorch import UNet_xy, UNet_z
from utils_pytorch import *
from optics_pytorch import codePattern
from train_vis import visualize_all

# GPU 设置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

######################################## Parameters ##################################################

DATA_PATH_root = './Depth_Dataset/'
results_dir = './FreeCam3D_model_pytorch/'
pattern_type = 'dotArray' # 'dotArray'
B = 1
B_sub = 1
N = 512
crop_train = False
crop_valid = False
H = 800
W = 1200
N_layers = 21
weight_reProj = 1e0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
is_conv_psf_train = False  # Whether to use PSF convolution in codePattern for training
is_conv_psf_valid = False  # Whether to use PSF convolution in codePattern for validation



def multi_rand_crop(img, crop_size, offset):
    """
    Random crop with given offsets
    Args:
        img: [B, H, W, C] tensor
        crop_size: size of crop
        offset: [B_sub, 2] offset positions
    Returns:
        [B_sub, crop_size, crop_size, C] cropped images
    """
    crops = []
    for i in range(offset.shape[0]):
        h_start = offset[i, 0]
        w_start = offset[i, 1]
        crop = img[0:1, h_start:h_start+crop_size, w_start:w_start+crop_size, :]
        crops.append(crop)
    return torch.cat(crops, dim=0)





def forward_model(z_p, z_c, pose_p2c, is_train=True, is_crop=True, is_conv_psf=True):
    """
    Forward model: generate synthetic camera image from depth maps
    """
    B = z_p.shape[0]
    
    # Convert depthmap to 3D point cloud
    coord_p = z2pointcloud(z_p)
    coord_c = z2pointcloud(z_c)
    
    # Generate pattern
    Ip = gen_pattern(B, H, W, N_layers, 0, pattern_type, 16)
    Ip_ref = withReflectance(Ip, z_p)
    
    # Generate coded projector pattern 
    Ip_coded = codePattern(Ip_ref, z_p, is_conv_psf=is_conv_psf) 
    
    # Warp pattern to camera view
    Ic, grid_p2c, grid_c2p = warp_p2c(Ip_coded, coord_p, coord_c, pose_p2c)
    
    # Add noise and scale
    Ic_exp = Ic + torch.rand(B, 1, 1, 1, device=device) * 0.05 + torch.randn(B, H, W, 1, device=device) * 0.005
    Ic_max = torch.max(torch.max(Ic_exp, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0]
    scale_factor = torch.rand(B, 1, 1, 1, device=device) * 0.3 + 0.7  # 0.7-1.0
    Ic_scaled = Ic_exp / Ic_max * scale_factor
    Ic_scaled = torch.clamp(Ic_scaled, 0.0, 1.0)
    
    # Get world coordinate x,y,z in camera view
    visible_mask_c_dense = gen_visible_mask(grid_p2c, 'all')
    
    # Get xy coordinates
    ray = pixel_to_ray_array()[:, :, 0:2]     
    xy_t = torch.from_numpy(ray).float()       # (H, W, 2) torch tensor
    xy = xy_t.unsqueeze(0).repeat(B, 1, 1, 1).float().to(device)
    
    
    xyz = torch.cat([xy, z_p], dim=-1)
    xyz_cView = bilinear_sampler(xyz, grid_c2p) * visible_mask_c_dense
    
    Ic_mask = ((Ic_scaled > 0.05).float()) * visible_mask_c_dense
    
    # # Random crop to sub batches
    # if is_train:
    #     offset = (torch.rand(B_sub, 2, device=device) * torch.tensor([[H - N - 1, W - N - 1]], 
    #                                                                    device=device)).long()
    #     Ic_scaled_crop = multi_rand_crop(Ic_scaled, N, offset)
    #     Ic_mask_crop = multi_rand_crop(Ic_mask, N, offset)
    #     xyz_cView_crop = multi_rand_crop(xyz_cView, N, offset)
    #     xy_crop = multi_rand_crop(xy, N, offset)
    #     z_c_crop = multi_rand_crop(z_c, N, offset)
    #     Ic_crop = multi_rand_crop(Ic, N, offset)
    # else:
    #     # For validation, use center crop or no crop
    #     Ic_scaled_crop = Ic_scaled
    #     Ic_mask_crop = Ic_mask
    #     xyz_cView_crop = xyz_cView
    #     xy_crop = xy
    #     z_c_crop = z_c
    #     Ic_crop = Ic
    if is_crop:
        offset = (torch.rand(B_sub, 2, device=device) * torch.tensor([[H - N - 1, W - N - 1]], 
                                                                    device=device)).long()
        Ic_scaled_crop = multi_rand_crop(Ic_scaled, N, offset)
        Ic_mask_crop = multi_rand_crop(Ic_mask, N, offset)
        xyz_cView_crop = multi_rand_crop(xyz_cView, N, offset)
        xy_crop = multi_rand_crop(xy, N, offset)
        z_c_crop = multi_rand_crop(z_c, N, offset)
        z_p_crop = multi_rand_crop(z_p, N, offset)
        Ic_crop = multi_rand_crop(Ic, N, offset)
    else:
        Ic_scaled_crop = Ic_scaled
        Ic_mask_crop = Ic_mask
        xyz_cView_crop = xyz_cView
        xy_crop = xy
        z_c_crop = z_c
        z_p_crop = z_p
        Ic_crop = Ic
    
    return Ic_scaled_crop, Ic_mask_crop, xyz_cView_crop, xy_crop, z_c_crop, z_p_crop, pose_p2c, z_p, Ic_crop, Ip_coded, Ic_scaled


def recon(model_xy, model_z, Ic_scaled_crop, Ic_mask_crop):
    """
    Reconstruction: predict xyz from camera image
    """
    Ic_scaled_crop = ln(Ic_scaled_crop, 17, 0.01)
    
    # xy/z two networks
    xy_cView_crop_hat = model_xy(Ic_scaled_crop)
    z_cView_crop_hat = model_z(Ic_scaled_crop)
    xyz_cView_crop_hat = torch.cat([xy_cView_crop_hat, z_cView_crop_hat], dim=-1)
    
    return xyz_cView_crop_hat


def reProj_IpGT(zp_cView, pose_p2c, xy, Ic, zcGT, maskC, IpGT):
    """
    Reprojection loss: warp pattern back and compare
    """
    # Expand mask
    maskC_expand = (torch.nn.functional.avg_pool2d(
        maskC.permute(0, 3, 1, 2), kernel_size=5, stride=1, padding=2
    ).permute(0, 2, 3, 1) > 0.999).float()
    
    zp_cView = zp_cView * maskC_expand
    B = pose_p2c.shape[0]
    B_all, N1, N2, _ = zp_cView.shape
    B_sub = B_all // B
    
    # Get grid_c2p
    zc = zp_cView_to_zc(zp_cView, pose_p2c, xy)
    ones = torch.ones(B_all, N1, N2, 1, device=zp_cView.device)
    ray = torch.cat([xy, ones], dim=-1)
    coord_c = points_in_camera_coords(zc, ray, allBatch=True).permute(0, 3, 1, 2)
    pose_c2p = torch.linalg.inv(pose_p2c).repeat(B_sub, 1, 1)
    grid_c2p = coord0TOgrid1(coord_c, pose_c2p)
    
    # Warp Ip to Ic
    Ic_reProj = bilinear_sampler(IpGT.repeat(B_sub, 1, 1, 1), grid_c2p)
    
    # Loss
    loss_zc_rms = cost_rms_mask(zcGT, zc, maskC_expand)
    loss_Ic_rms = cost_rms_mask(Ic, Ic_reProj, maskC_expand)
    loss_Ic_L1 = cost_L1_mask(Ic, Ic_reProj, maskC_expand)
    
    loss = loss_Ic_L1 + loss_zc_rms
    
    return loss, loss_zc_rms, loss_Ic_rms, loss_Ic_L1


def cal_cost(xyz_cView_crop, xyz_cView_crop_hat, Ic_mask_crop):
    """
    Calculate reconstruction loss
    """
    loss_xy_rms = cost_rms_mask(xyz_cView_crop[:, :, :, :2], xyz_cView_crop_hat[:, :, :, :2], Ic_mask_crop)
    loss_z_rms = cost_rms_mask(xyz_cView_crop[:, :, :, 2:3], xyz_cView_crop_hat[:, :, :, 2:3], Ic_mask_crop)
    
    # Expand mask for gradient loss
    Ic_mask_crop_expand = (torch.nn.functional.avg_pool2d(
        Ic_mask_crop.permute(0, 3, 1, 2), kernel_size=5, stride=1, padding=2
    ).permute(0, 2, 3, 1) > 0.999).float()
    
    loss_z_grad = cost_grad_mask(xyz_cView_crop[:, :, :, 2:3], xyz_cView_crop_hat[:, :, :, 2:3], Ic_mask_crop_expand)
    
    loss = loss_xy_rms + loss_z_rms + loss_z_grad
    
    return loss, loss_xy_rms, loss_z_rms, loss_z_grad


def main():
    # Create models
    model_xy = UNet_xy().to(device)
    model_z = UNet_z().to(device)
    
    # Optimizer
    lr_val = 1e-4
    optimizer = optim.Adam(list(model_xy.parameters()) + list(model_z.parameters()), lr=lr_val)
    
    # Data loaders
    train_loader = read_data(DATA_PATH_root, B, mode='train')
    valid_loader = read_data(DATA_PATH_root, B, mode='valid')
    
    # TensorBoard writer
    os.makedirs(results_dir, exist_ok=True)
    writer = SummaryWriter(results_dir + '/summary/')
    
    # Load checkpoint if exists
    best_loss = 100.0
    start_iter = 0
    checkpoint_path = os.path.join(results_dir, 'checkpoint.pth')
    best_model_path = os.path.join(results_dir, 'best_model.pth')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model_xy.load_state_dict(checkpoint['model_xy'])
        model_z.load_state_dict(checkpoint['model_z'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint['iteration']
        best_loss = checkpoint.get('best_loss', 100.0)
        print(f'Continue training from iteration {start_iter}, best loss = {best_loss:.6f}')
    else:
        print(f'Start training, save to: {results_dir}')
    
    # Training loop
    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)
    
    for i in range(start_iter, 1000000):
        # Get training batch
        try:
            train_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            train_batch = next(train_iter)
        
        z_p_train = train_batch['z_p'].to(device)
        z_c_train = train_batch['z_c'].to(device)
        pose_p2c_train = train_batch['pose_p2c'].to(device)
        
        # Forward pass
        model_xy.train()
        model_z.train()
        
        Ic_scaled_crop, Ic_mask_crop, xyz_cView_crop, xy_crop, z_c_crop, z_p_crop, pose_p2c, z_p, Ic_crop, Ip_coded, Ic_scaled = \
            forward_model(z_p_train, z_c_train, pose_p2c_train, is_train=True, is_crop=crop_train, is_conv_psf=is_conv_psf_train)
        
        # Reconstruction
        xyz_cView_crop_hat = recon(model_xy, model_z, Ic_scaled_crop, Ic_mask_crop)
        
        # Calculate loss
        loss_recon, loss_xy_rms, loss_z_rms, loss_z_grad = cal_cost(xyz_cView_crop, xyz_cView_crop_hat, Ic_mask_crop)
        loss_reproj, loss_zc_rms, loss_Ic_rms, loss_Ic_L1 = reProj_IpGT(
            xyz_cView_crop_hat[:, :, :, 2:3], pose_p2c, xy_crop, Ic_crop, z_c_crop, Ic_mask_crop, Ip_coded
        )
        
        loss_train = loss_recon + loss_reproj * weight_reProj
        
        # Backward pass
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        # Training visualization (less frequent to save I/O)
        if i % 1000 == 0:
            print(f"Saving training visualizations at iter {i}...")
            print(f"  Train Loss = {loss_train.item():.6f} (recon: {loss_recon.item():.6f}, reproj: {loss_reproj.item():.6f})")
            with torch.no_grad():
                vis_outdir = os.path.join(results_dir, 'recon')
                tb_images_train = visualize_all(
                    xyz_pred=xyz_cView_crop_hat.detach(),
                    z_p=z_p,
                    z_c=z_c_train,
                    Ic_scaled=Ic_scaled,
                    Ip_coded=Ip_coded,
                    mask=Ic_mask_crop,
                    xyz_gt=xyz_cView_crop,
                    z_p_crop=z_p_crop,
                    z_c_crop=z_c_crop,
                    Ic_scaled_crop=Ic_scaled_crop,
                    outdir=vis_outdir,
                    prefix='train',
                    idx=i,
                    max_samples=2,  # Save fewer samples for training to save space
                    save_to_disk=True
                )
                
                # Add training images to TensorBoard
                for img_name, img_tensor in tb_images_train.items():
                    writer.add_image(img_name, img_tensor, i)
        
        # Validation and logging
        if i % 5000 == 0:
            model_xy.eval()
            model_z.eval()
            
            with torch.no_grad():
                # Get validation batch
                try:
                    valid_batch = next(valid_iter)
                except StopIteration:
                    valid_iter = iter(valid_loader)
                    valid_batch = next(valid_iter)
                
                z_p_valid = valid_batch['z_p'].to(device)
                z_c_valid = valid_batch['z_c'].to(device)
                pose_p2c_valid = valid_batch['pose_p2c'].to(device)
                
                Ic_scaled_crop_v, Ic_mask_crop_v, xyz_cView_crop_v, xy_crop_v, z_c_crop_v, z_p_crop_v, pose_p2c_v, z_p_v, Ic_crop_v, Ip_coded_v, Ic_scaled_v = \
                    forward_model(z_p_valid, z_c_valid, pose_p2c_valid, is_train=False, is_crop=crop_valid, is_conv_psf=is_conv_psf_valid)
                
                xyz_cView_crop_hat_v = recon(model_xy, model_z, Ic_scaled_crop_v, Ic_mask_crop_v)
                
                loss_recon_v, loss_xy_rms_v, loss_z_rms_v, loss_z_grad_v = cal_cost(xyz_cView_crop_v, xyz_cView_crop_hat_v, Ic_mask_crop_v)
                loss_reproj_v, loss_zc_rms_v, loss_Ic_rms_v, loss_Ic_L1_v = reProj_IpGT(
                    xyz_cView_crop_hat_v[:, :, :, 2:3], pose_p2c_v, xy_crop_v, Ic_crop_v, z_c_crop_v, Ic_mask_crop_v, Ip_coded_v
                )
                
                loss_valid = loss_recon_v + loss_reproj_v * weight_reProj
                
                print(f"Iter {i}, Valid Loss = {loss_valid.item():.6f} (recon: {loss_recon_v.item():.6f}, reproj: {loss_reproj_v.item():.6f})")
                
                # Visualizations during validation
                vis_outdir = os.path.join(results_dir, 'recon')
                tb_images = visualize_all(
                    xyz_pred=xyz_cView_crop_hat_v,
                    z_p=z_p_valid,
                    z_c=z_c_valid,
                    Ic_scaled=Ic_scaled_v,
                    Ip_coded=Ip_coded_v,
                    mask=Ic_mask_crop_v,
                    xyz_gt=xyz_cView_crop_v,
                    z_p_crop=z_p_crop_v,
                    z_c_crop=z_c_crop_v,
                    Ic_scaled_crop=Ic_scaled_crop_v,
                    outdir=vis_outdir,
                    prefix='valid',
                    idx=i,
                    max_samples=3,
                    save_to_disk=True
                )
                
                # Add images to TensorBoard
                for img_name, img_tensor in tb_images.items():
                    writer.add_image(img_name, img_tensor, i)
            
            print(f"Iter {i}, Loss = {loss_valid.item():.6f}")
            
            # TensorBoard logging
            writer.add_scalar('Loss/train', loss_train.item(), i)
            writer.add_scalar('Loss/valid', loss_valid.item(), i)
            writer.add_scalar('Loss/xy_rms', loss_xy_rms.item(), i)
            writer.add_scalar('Loss/z_rms', loss_z_rms.item(), i)
            writer.add_scalar('Loss/z_grad', loss_z_grad.item(), i)
            
            # Save checkpoint
            torch.save({
                'iteration': i,
                'model_xy': model_xy.state_dict(),
                'model_z': model_z.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
            }, checkpoint_path)
            
            # Save best model
            if loss_valid.item() < best_loss and i > 1:
                best_loss = loss_valid.item()
                torch.save({
                    'iteration': i,
                    'model_xy': model_xy.state_dict(),
                    'model_z': model_z.state_dict(),
                    'best_loss': best_loss,
                }, best_model_path)
                print(f'Best model saved at iter {i} with loss = {best_loss:.6f}')
    
    writer.close()


if __name__ == '__main__':
    main()

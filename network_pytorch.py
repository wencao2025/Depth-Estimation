'''
Neural networks for estimation - PyTorch version
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


z_min = 0.695714566
z_max = 0.941062289
XY_min = np.array([-768/2/1666.67, -512/2/1666.67], dtype=np.float32)
XY_max = np.array([768/2/1666.67, 512/2/1666.67], dtype=np.float32)


class ConvBNReLU(nn.Module):
    """
    Convolution + BatchNorm + ReLU with symmetric padding
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, relu=True):
        super(ConvBNReLU, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)  # PyTorch momentum = 1 - TF momentum
        self.relu = nn.ReLU(inplace=True) if relu else None
    
    def forward(self, x):
        # Symmetric padding (reflect mode in PyTorch is similar to symmetric in TF)
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class UpConv(nn.Module):
    """
    Upsampling (nearest neighbor) + Convolution with symmetric padding
    """
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
        self.padding = 1
    
    def forward(self, x):
        # Resize using nearest neighbor interpolation
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # Symmetric padding
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        x = self.conv(x)
        return x


class UNet_xy(nn.Module):
    """
    UNet for predicting x/z and y/z coordinates
    """
    def __init__(self):
        super(UNet_xy, self).__init__()
        
        # Encoder (downsampling path)
        self.down1_1 = ConvBNReLU(1, 32)
        self.down1_2 = ConvBNReLU(32, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2_1 = ConvBNReLU(32, 64)
        self.down2_2 = ConvBNReLU(64, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.down3_1 = ConvBNReLU(64, 128)
        self.down3_2 = ConvBNReLU(128, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.down4_1 = ConvBNReLU(128, 256)
        self.down4_2 = ConvBNReLU(256, 256)
        self.pool4 = nn.MaxPool2d(2)
        
        self.down5_1 = ConvBNReLU(256, 512)
        self.down5_2 = ConvBNReLU(512, 512)
        
        # Decoder (upsampling path)
        self.up4_0 = UpConv(512, 256)
        self.up4_1 = ConvBNReLU(512, 256)  # 256 + 256 from skip connection
        self.up4_2 = ConvBNReLU(256, 256)
        
        self.up3_0 = UpConv(256, 128)
        self.up3_1 = ConvBNReLU(256, 128)  # 128 + 128 from skip connection
        self.up3_2 = ConvBNReLU(128, 128)
        
        self.up2_0 = UpConv(128, 64)
        self.up2_1 = ConvBNReLU(128, 64)  # 64 + 64 from skip connection
        self.up2_2 = ConvBNReLU(64, 64)
        
        self.up1_0 = UpConv(64, 32)
        self.up1_1 = ConvBNReLU(64, 32)  # 32 + 32 from skip connection
        self.up1_2 = ConvBNReLU(32, 32)
        
        # Final output layer (1x1 conv)
        self.up1_3 = ConvBNReLU(32, 2, kernel_size=1, relu=False)
        
        # Register XY_min and XY_max as buffers (not parameters)
        self.register_buffer('xy_min', torch.from_numpy(XY_min).reshape(1, 2, 1, 1))
        self.register_buffer('xy_max', torch.from_numpy(XY_max).reshape(1, 2, 1, 1))
    
    def forward(self, x):
        # Input: [B, H, W, 1] -> convert to [B, 1, H, W]
        x = x.permute(0, 3, 1, 2)
        
        # Encoder
        down1 = self.down1_2(self.down1_1(x))
        down2 = self.down2_2(self.down2_1(self.pool1(down1)))
        down3 = self.down3_2(self.down3_1(self.pool2(down2)))
        down4 = self.down4_2(self.down4_1(self.pool3(down3)))
        down5 = self.down5_2(self.down5_1(self.pool4(down4)))
        
        # Decoder with skip connections
        up4 = self.up4_2(self.up4_1(torch.cat([self.up4_0(down5), down4], dim=1)))
        up3 = self.up3_2(self.up3_1(torch.cat([self.up3_0(up4), down3], dim=1)))
        up2 = self.up2_2(self.up2_1(torch.cat([self.up2_0(up3), down2], dim=1)))
        up1 = self.up1_2(self.up1_1(torch.cat([self.up1_0(up2), down1], dim=1)))
        
        # Final output
        out = torch.sigmoid(self.up1_3(up1))  # [B, 2, H, W], range [0, 1]
        
        # Scale to XY range
        out = out * (self.xy_max - self.xy_min) + self.xy_min
        
        # Convert back to [B, H, W, 2]
        out = out.permute(0, 2, 3, 1)
        return out


class UNet_z(nn.Module):
    """
    UNet for predicting z (depth) coordinate
    """
    def __init__(self):
        super(UNet_z, self).__init__()
        
        # Encoder (downsampling path)
        self.down1_1 = ConvBNReLU(1, 32)
        self.down1_2 = ConvBNReLU(32, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2_1 = ConvBNReLU(32, 64)
        self.down2_2 = ConvBNReLU(64, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.down3_1 = ConvBNReLU(64, 128)
        self.down3_2 = ConvBNReLU(128, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.down4_1 = ConvBNReLU(128, 256)
        self.down4_2 = ConvBNReLU(256, 256)
        self.pool4 = nn.MaxPool2d(2)
        
        self.down5_1 = ConvBNReLU(256, 512)
        self.down5_2 = ConvBNReLU(512, 512)
        
        # Decoder (upsampling path)
        self.up4_0 = UpConv(512, 256)
        self.up4_1 = ConvBNReLU(512, 256)
        self.up4_2 = ConvBNReLU(256, 256)
        
        self.up3_0 = UpConv(256, 128)
        self.up3_1 = ConvBNReLU(256, 128)
        self.up3_2 = ConvBNReLU(128, 128)
        
        self.up2_0 = UpConv(128, 64)
        self.up2_1 = ConvBNReLU(128, 64)
        self.up2_2 = ConvBNReLU(64, 64)
        
        self.up1_0 = UpConv(64, 32)
        self.up1_1 = ConvBNReLU(64, 32)
        self.up1_2 = ConvBNReLU(32, 32)
        
        # Final output layer (1x1 conv, output 1 channel for z)
        self.up1_3 = ConvBNReLU(32, 1, kernel_size=1, relu=False)
        
        # Z range
        self.z_min = z_min
        self.z_max = z_max
    
    def forward(self, x):
        # Input: [B, H, W, 1] -> convert to [B, 1, H, W]
        x = x.permute(0, 3, 1, 2)
        
        # Encoder
        down1 = self.down1_2(self.down1_1(x))
        down2 = self.down2_2(self.down2_1(self.pool1(down1)))
        down3 = self.down3_2(self.down3_1(self.pool2(down2)))
        down4 = self.down4_2(self.down4_1(self.pool3(down3)))
        down5 = self.down5_2(self.down5_1(self.pool4(down4)))
        
        # Decoder with skip connections
        up4 = self.up4_2(self.up4_1(torch.cat([self.up4_0(down5), down4], dim=1)))
        up3 = self.up3_2(self.up3_1(torch.cat([self.up3_0(up4), down3], dim=1)))
        up2 = self.up2_2(self.up2_1(torch.cat([self.up2_0(up3), down2], dim=1)))
        up1 = self.up1_2(self.up1_1(torch.cat([self.up1_0(up2), down1], dim=1)))
        
        # Final output
        out = torch.sigmoid(self.up1_3(up1))  # [B, 1, H, W], range [0, 1]
        
        # Scale to z range
        out = out * (self.z_max - self.z_min) + self.z_min
        
        # Convert back to [B, H, W, 1]
        out = out.permute(0, 2, 3, 1)
        return out


# 使用示例
if __name__ == '__main__':
    # 测试网络
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model_xy = UNet_xy().to(device)
    model_z = UNet_z().to(device)
    
    # 测试输入
    batch_size = 2
    H, W = 256, 256
    x = torch.randn(batch_size, H, W, 1).to(device)
    
    # 前向传播
    model_xy.eval()
    model_z.eval()
    with torch.no_grad():
        out_xy = model_xy(x)
        out_z = model_z(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output XY shape: {out_xy.shape}")
    print(f"Output Z shape: {out_z.shape}")
    print(f"XY range: [{out_xy.min().item():.4f}, {out_xy.max().item():.4f}]")
    print(f"Z range: [{out_z.min().item():.4f}, {out_z.max().item():.4f}]")
    
    # 统计参数量
    total_params_xy = sum(p.numel() for p in model_xy.parameters())
    total_params_z = sum(p.numel() for p in model_z.parameters())
    print(f"\nTotal parameters in UNet_xy: {total_params_xy:,}")
    print(f"Total parameters in UNet_z: {total_params_z:,}")
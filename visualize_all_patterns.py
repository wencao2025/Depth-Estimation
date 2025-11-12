"""
Visualize all pattern modes and save as GIF animations
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import os
import numpy as np
from utils_pytorch import gen_pattern

# 创建输出目录
os.makedirs('pattern_visualization', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 参数设置
B = 1
H, W = 800, 1200
N_layers = 21
grid_Isigma = 0.1
stride = 16

# 所有可用的模式
pattern_modes = ['grid', 'kronTwoFix']

# 如果有 kinect 和 MArray 的数据文件，也可以包含
if os.path.exists('ProjPatterns/kinect1200.mat'):
    pattern_modes.append('kinect')
if os.path.exists('ProjPatterns/M_array.mat'):
    pattern_modes.append('MArray')

print(f"\nGenerating patterns for modes: {pattern_modes}")

# 为每种模式生成 GIF
for mode in pattern_modes:
    print(f"\n{'='*60}")
    print(f"Processing mode: {mode}")
    print(f"{'='*60}")
    
    try:
        # 生成图案
        grid = gen_pattern(B, H, W, N_layers, grid_Isigma, mode, stride, device)
        print(f"✓ Generated grid shape: {grid.shape}")
        print(f"  Value range: [{grid.min().item():.4f}, {grid.max().item():.4f}]")
        
        # 1. 保存单层静态图片（第10层）
        img = grid[0, :, :, 10].cpu().numpy()
        plt.figure(figsize=(12, 8))
        plt.imshow(img, cmap='gray')
        plt.colorbar(label='Intensity')
        plt.title(f'{mode} Pattern - Layer 10', fontsize=16)
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.savefig(f'pattern_visualization/{mode}_layer10.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {mode}_layer10.png")
        
        # 2. 保存所有层的 GIF 动画
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def animate(layer):
            ax.clear()
            img = grid[0, :, :, layer].cpu().numpy()
            im = ax.imshow(img, cmap='gray', vmin=grid.min().item(), vmax=grid.max().item())
            ax.set_title(f'{mode} Pattern - Layer {layer}/{N_layers-1}', fontsize=16)
            ax.set_xlabel('Width (pixels)')
            ax.set_ylabel('Height (pixels)')
            # 只在第一帧添加 colorbar
            if layer == 0:
                plt.colorbar(im, ax=ax, label='Intensity')
            return [im]
        
        print(f"  Creating GIF animation...")
        ani = animation.FuncAnimation(fig, animate, frames=N_layers, interval=150, blit=True, repeat=True)
        ani.save(f'pattern_visualization/{mode}_animation.gif', writer='pillow', fps=7)
        plt.close()
        print(f"✓ Saved: {mode}_animation.gif")
        
        # 3. 保存局部放大的 GIF
        center_h, center_w = 400, 600
        crop_size = 200
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def animate_detail(layer):
            ax.clear()
            img = grid[0, 
                      center_h-crop_size//2:center_h+crop_size//2,
                      center_w-crop_size//2:center_w+crop_size//2,
                      layer].cpu().numpy()
            im = ax.imshow(img, cmap='gray', interpolation='nearest')
            ax.set_title(f'{mode} Pattern Detail - Layer {layer}/{N_layers-1}', fontsize=14)
            ax.axis('off')
            return [im]
        
        print(f"  Creating detail GIF animation...")
        ani_detail = animation.FuncAnimation(fig, animate_detail, frames=N_layers, interval=150, blit=True, repeat=True)
        ani_detail.save(f'pattern_visualization/{mode}_detail_animation.gif', writer='pillow', fps=7)
        plt.close()
        print(f"✓ Saved: {mode}_detail_animation.gif")
        
        # 4. 保存多层对比图
        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        selected_layers = [0, 4, 8, 12, 16, 20]
        
        for idx, layer in enumerate(selected_layers):
            row, col = idx // 3, idx % 3
            img = grid[0, :, :, layer].cpu().numpy()
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f'Layer {layer}', fontsize=14)
            axes[row, col].axis('off')
        
        plt.suptitle(f'{mode} Pattern - Multiple Layers', fontsize=18)
        plt.tight_layout()
        plt.savefig(f'pattern_visualization/{mode}_layers_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {mode}_layers_comparison.png")
        
    except Exception as e:
        print(f"❌ Error processing mode '{mode}': {e}")
        continue

# 创建对比所有模式的图片
print(f"\n{'='*60}")
print("Creating comparison of all patterns...")
print(f"{'='*60}")

fig, axes = plt.subplots(len(pattern_modes), 3, figsize=(18, 6*len(pattern_modes)))
if len(pattern_modes) == 1:
    axes = axes.reshape(1, -1)

for idx, mode in enumerate(pattern_modes):
    try:
        grid = gen_pattern(B, H, W, N_layers, grid_Isigma, mode, stride, device)
        
        # 显示 3 个不同的层
        for col, layer in enumerate([0, 10, 20]):
            img = grid[0, :, :, layer].cpu().numpy()
            axes[idx, col].imshow(img, cmap='gray')
            if col == 0:
                axes[idx, col].set_ylabel(f'{mode}', fontsize=14, rotation=0, ha='right', va='center')
            axes[idx, col].set_title(f'Layer {layer}', fontsize=12)
            axes[idx, col].axis('off')
    except:
        pass

plt.suptitle('Pattern Comparison - Different Modes and Layers', fontsize=18)
plt.tight_layout()
plt.savefig('pattern_visualization/all_patterns_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: all_patterns_comparison.png")

# 创建摘要
print(f"\n{'='*60}")
print("✅ All visualizations completed!")
print(f"{'='*60}")
print(f"\nFiles saved in: pattern_visualization/")
print("\nGenerated files:")

for mode in pattern_modes:
    print(f"\n{mode} mode:")
    print(f"  - {mode}_layer10.png (static image)")
    print(f"  - {mode}_animation.gif (full view)")
    print(f"  - {mode}_detail_animation.gif (zoomed in)")
    print(f"  - {mode}_layers_comparison.png (multi-layer comparison)")

print(f"\nComparison:")
print(f"  - all_patterns_comparison.png (all modes side by side)")

# 统计信息
total_files = len(pattern_modes) * 4 + 1
print(f"\n📊 Total files created: {total_files}")
print(f"📁 Directory: {os.path.abspath('pattern_visualization/')}")

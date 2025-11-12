
# 直接定义并立即调用（处理 requires_grad 的 tensor）
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

# 立即尝试调用（如果变量不存在会捕获 NameError 并提示）
try:
    # example: save at 2 FPS and include per-frame labels
    save_three_panel_gif(xyz_cView_crop_hat, out_path='./vis_debug_pattern/xyz_three_panel.gif', normalize='per_channel', fps=2.0)
except NameError:
    print("变量 `xyz_cView_crop_hat` 在当前作用域中未找到。请确保在 Debug Console 的当前断点/作用域中执行此代码，或把变量名改成你实际使用的名字。")
except Exception as e:
    print("运行时出错：", repr(e))
    raise


############################################################################





import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import imageio

# 创建保存目录
save_dir = './vis_debug_pattern'
os.makedirs(save_dir, exist_ok=True)

def tensor_to_numpy(tensor):
    """转换 tensor 到 numpy，处理各种情况"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)

def normalize_for_vis(data):
    """归一化到 [0, 255] uint8"""
    data = data.astype(np.float32)
    data_min, data_max = data.min(), data.max()
    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min) * 255
    else:
        data = np.zeros_like(data)
    return data.astype(np.uint8)

def save_png(data, filepath, name):
    """保存单张 PNG"""
    data_np = tensor_to_numpy(data)
    
    # 取第一个 batch（如果有）
    while data_np.ndim > 3:
        data_np = data_np[0]
    
    # 移除单通道维度
    if data_np.ndim == 3 and data_np.shape[-1] == 1:
        data_np = data_np[..., 0]
    elif data_np.ndim == 3 and data_np.shape[0] == 1:
        data_np = data_np[0]
    
    data_norm = normalize_for_vis(data_np)
    img = Image.fromarray(data_norm)
    img.save(filepath)
    print(f"✓ Saved {name}: {filepath}")

def save_gif_with_labels(data, filepath, name, label_prefix="Frame"):
    """保存带标题的 GIF 动画，按最后一个维度生成帧"""
    data_np = tensor_to_numpy(data)
    
    # 取第一个 batch（如果第一维度是 batch）
    if data_np.ndim == 5:  # [B, H, W, C, D] 或 [B, H, W, D, C]
        data_np = data_np[0]
    elif data_np.ndim == 4:  # [B, H, W, D] 或 [H, W, C, D]
        # 如果第一维度看起来像 batch size（通常 < 10）
        if data_np.shape[0] <= 10 and data_np.shape[0] < data_np.shape[-1]:
            data_np = data_np[0]  # 去掉 batch
    
    # 现在应该是 [H, W, D] 或 [H, W, C, D] 格式
    # 最后一个维度是深度层数
    num_frames = data_np.shape[-1]
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    frames = []
    for i in range(num_frames):
        # 按最后一个维度切片
        frame_data = data_np[..., i]  # [H, W] 或 [H, W, C]
        
        # 移除单通道维度
        if frame_data.ndim == 3 and frame_data.shape[-1] == 1:
            frame_data = frame_data[..., 0]
        
        frame_norm = normalize_for_vis(frame_data)
        img = Image.fromarray(frame_norm).convert('RGB')
        
        # 添加文字标签
        draw = ImageDraw.Draw(img)
        text = f"{label_prefix} {i+1}/{num_frames}"
        
        # 绘制文字背景（黑色矩形）
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.rectangle([(5, 5), (15 + text_width, 15 + text_height)], fill='black')
        
        # 绘制白色文字
        draw.text((10, 10), text, fill='white', font=font)
        
        frames.append(np.array(img))
    
    imageio.mimsave(filepath, frames, fps=2, loop=0)
    print(f"✓ Saved {name}: {filepath} ({num_frames} frames)")


def save_gif_firstdim(data, filepath, name, label_prefix="Frame"):
    """保存带标题的 GIF 动画，按第一个维度生成帧（batch as frames）"""
    data_np = tensor_to_numpy(data)

    # 如果数据没有 batch 维，把最后一个维度当帧（回退）
    if data_np.ndim == 3:
        # [H, W, D] -> use last dim as frames
        num_frames = data_np.shape[-1]
        use_last = True
    else:
        # Use first dim as frames
        num_frames = data_np.shape[0]
        use_last = False

    # 尝试加载字体
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

        # 如果是 [..., C] 且 C==1，移除通道
        if frame_data.ndim == 3 and frame_data.shape[-1] == 1:
            frame_data = frame_data[..., 0]

        # 如果是 [1, H, W] 或 [1, H, W, C]
        if frame_data.ndim == 4 and frame_data.shape[0] == 1:
            frame_data = frame_data[0]

        frame_norm = normalize_for_vis(frame_data)
        img = Image.fromarray(frame_norm).convert('RGB')

        # 添加文字标签
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

# ===== 保存 PNG 图像 =====
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

# 6. Ip_coded (保存为 PNG)
if 'Ip_coded' in locals() or 'Ip_coded' in globals():
    save_png(Ip_coded, f'{save_dir}/Ip_coded.png', 'Ip_coded')

if 'Ic_mask' in locals() or 'Ic_mask' in globals():
    save_png(Ic_mask, f'{save_dir}/Ic_mask.png', 'Ic_mask')

# ===== 保存 GIF 动画 =====
print("\n=== Saving GIF animations ===")

# 7. Ip (按最后一个维度)
if 'Ip' in locals() or 'Ip' in globals():
    save_gif_with_labels(Ip, f'{save_dir}/Ip.gif', 'Ip', label_prefix="Depth")

# 8. Ip_ref (按最后一个维度，保存为 GIF)
if 'Ip_ref' in locals() or 'Ip_ref' in globals():
    save_gif_with_labels(Ip_ref, f'{save_dir}/Ip_ref.gif', 'Ip_ref', label_prefix="Depth")

# ===== 保存 batch-as-frames GIF（按第一个维度） =====
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

# 变量名（若你的变量名不同，改这里）
candidates = ['psfs', 'PSFs', 'psf', 'PSF']
psf_var = None
for n in candidates:
    if n in globals() or n in locals():
        psf_var = globals().get(n, locals().get(n))
        break
if psf_var is None:
    raise RuntimeError("找不到 PSFs 变量，请在 Debug Console 作用域内定义名为 psfs/PSFs/psf/PSF 的变量，或修改此 snippet 的候选名。")

# 输出目录
out_dir = './vis_debug_pattern/psfs'
os.makedirs(out_dir, exist_ok=True)
gif_path = os.path.join(out_dir, 'psfs_firstdim.gif')

# 转成 numpy（支持 torch.Tensor 在 GPU 上）
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

arr = to_numpy(psf_var)

# 支持形状：
# [N, H, W], [N, H, W, 1], [N, H, W, C], 也支持 leading batch [1, N, H, W] 等 -> 会尝试找到第一个维度为帧
if arr.ndim >= 4 and arr.shape[0] == 1 and arr.ndim == 4:
    # 例如 [1, N, H, W] -> 变为 [N, H, W]
    arr = arr[0]

# 如果是 [H,W,D]（最后一维是 depth），我们想按第一个维度（batch）保存，这里不能处理这种情况——期待第一维为帧或 N,H,W
if arr.ndim == 3:
    # 视为 [N, H, W] 或 [H, W, N] 无法确定：我们优先认为是 [N,H,W] (第一个维度是帧)
    n_frames = arr.shape[0]
    has_chan = False
elif arr.ndim == 4:
    # [N, H, W, C] 或 [N, C, H, W] 的可能性，优先认为是 [N, H, W, C]
    # 若认为是 [N, C, H, W]，请先在 Debug Console 里转置
    n_frames = arr.shape[0]
    has_chan = True
else:
    raise RuntimeError(f"不支持的 PSF 数组形状: {arr.shape}. 期望 [N,H,W] 或 [N,H,W,C].")

# 是否跨帧统一归一化（True：所有帧使用全局 min/max；False：每帧独立归一化）
unify_norm = True

if unify_norm:
    g_min = float(arr.min())
    g_max = float(arr.max())
else:
    g_min = g_max = None

# 字体
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except:
    font = ImageFont.load_default()

frames = []
for i in range(n_frames):
    frame = arr[i]
    # 如果有 channel dim 且 C==1，降为灰度
    if has_chan and frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[..., 0]
    # 归一化到 uint8
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
    # 添加标签
    # draw = ImageDraw.Draw(img)
    # text = f"PSF {i+1}/{n_frames}"
    # bbox = draw.textbbox((0,0), text, font=font)
    # tw = bbox[2]-bbox[0]; th = bbox[3]-bbox[1]
    # draw.rectangle([(5,5),(10+tw,10+th)], fill='black')
    # draw.text((8,7), text, fill='white', font=font)
    frames.append(np.array(img))
    # 也把每帧单独保存为 PNG（可选）
    png_path = os.path.join(out_dir, f'psf_{i+1:03d}.png')
    Image.fromarray(f_u8).save(png_path)

# 保存 GIF（fps 可改）
imageio.mimsave(gif_path, frames, fps=2, loop=0)
print(f"Saved PSF GIF -> {gif_path} and {n_frames} PNGs in {out_dir}")


############################################################################
##########################save pattern vis code below##########################
############################################################################
# 运行前提：确保能访问到项目的 import 路径（通常在项目根目录运行调试即可）。
# 说明：此 snippet 会尝试从当前作用域或常见模块导入 gen_pattern，
#       然后为每个 patternMode 生成 pattern 并把最后一个维度作为 GIF 的帧保存。

import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
import torch

save_base = './vis_debug_pattern/patterns'
os.makedirs(save_base, exist_ok=True)

# 1) 找到 gen_pattern 函数（优先当前作用域，否则尝试导入常见模块）
if 'gen_pattern' in globals() or 'gen_pattern' in locals():
    gen_pattern_fn = gen_pattern
else:
    # 尝试从训练/工具模块导入
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

# 2) helper：把 tensor/ndarray 按最后一个维度存为 GIF（每帧带标签）
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class StereoDepthEstimator:
    """
    Traditional stereo vision triangulation-based depth estimation
    """
    def __init__(self, focal_length, baseline, img_width, img_height):
        """
        Args:
            focal_length: Focal length in pixels
            baseline: Baseline distance in meters
            img_width: Image width
            img_height: Image height
        """
        self.focal_length = focal_length
        self.baseline = baseline
        self.img_width = img_width
        self.img_height = img_height
        
    def compute_disparity_sgbm(self, left_img, right_img, 
                                min_disparity=0, 
                                num_disparities=128,
                                block_size=5):
        """
        Compute disparity map using Semi-Global Block Matching
        
        Args:
            left_img: Left image (H, W, 3) numpy array
            right_img: Right image (H, W, 3) numpy array
            min_disparity: Minimum disparity
            num_disparities: Disparity range (must be divisible by 16)
            block_size: Matching block size (odd number, 3-11)
            
        Returns:
            disparity: Disparity map (H, W) torch tensor
        """
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        # Create SGBM object
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        # Convert to torch tensor
        disparity_torch = torch.from_numpy(disparity)
        
        return disparity_torch
    
    def disparity_to_depth(self, disparity, min_depth=0.1, max_depth=100.0):
        """
        Convert disparity map to depth map using triangulation formula
        
        Args:
            disparity: Disparity map (H, W) torch tensor
            min_depth: Minimum depth threshold in meters
            max_depth: Maximum depth threshold in meters
            
        Returns:
            depth: Depth map (H, W) torch tensor
        """
        # Avoid division by zero
        disparity = torch.clamp(disparity, min=1e-6)
        
        # Depth = (baseline × focal_length) / disparity
        depth = (self.baseline * self.focal_length) / disparity
        
        # Clamp depth to valid range
        depth = torch.clamp(depth, min=min_depth, max=max_depth)
        
        return depth
    
    def point_cloud_from_depth(self, depth, K=None):
        """
        Generate 3D point cloud from depth map
        
        Args:
            depth: Depth map (H, W) torch tensor
            K: Camera intrinsic matrix (3, 3), if None use simplified model
            
        Returns:
            points_3d: Point cloud (H, W, 3) - (X, Y, Z)
        """
        H, W = depth.shape
        
        # Create pixel coordinate grid
        u = torch.arange(0, W, dtype=torch.float32)
        v = torch.arange(0, H, dtype=torch.float32)
        u, v = torch.meshgrid(u, v, indexing='xy')
        
        if K is None:
            # Simplified model: assume principal point at image center
            cx = W / 2.0
            cy = H / 2.0
            fx = fy = self.focal_length
        else:
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
        
        # Triangulation back-projection
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = depth
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        
        # Stack into point cloud
        points_3d = torch.stack([X, Y, Z], dim=-1)
        
        return points_3d


class CostVolumeDepthEstimator(nn.Module):
    """
    Cost Volume-based depth estimation network
    Combines traditional triangulation with deep learning
    """
    def __init__(self, max_disparity=192):
        super().__init__()
        self.max_disparity = max_disparity
        
        # Feature extraction network
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(inplace=True),
        )
        
        # Cost volume aggregation
        self.cost_aggregation = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, 3, 1, 1),
        )
        
    def build_cost_volume(self, left_feat, right_feat):
        """
        Build cost volume for stereo matching
        """
        B, C, H, W = left_feat.shape
        cost_volume = torch.zeros(B, C, self.max_disparity // 4, H, W, 
                                   device=left_feat.device)
        
        for d in range(self.max_disparity // 4):
            if d == 0:
                cost_volume[:, :, d, :, :] = left_feat - right_feat
            else:
                cost_volume[:, :, d, :, d:] = left_feat[:, :, :, d:] - right_feat[:, :, :, :-d]
        
        return cost_volume
    
    def forward(self, left_img, right_img):
        """
        Forward pass
        
        Args:
            left_img: (B, 3, H, W)
            right_img: (B, 3, H, W)
            
        Returns:
            disparity: (B, 1, H, W)
        """
        # Feature extraction
        left_feat = self.feature_extractor(left_img)
        right_feat = self.feature_extractor(right_img)
        
        # Build cost volume
        cost_volume = self.build_cost_volume(left_feat, right_feat)
        
        # Aggregate costs
        cost = self.cost_aggregation(cost_volume)  # (B, 1, D, H, W)
        cost = cost.squeeze(1)  # (B, D, H, W)
        
        # Soft argmin for disparity regression
        prob = F.softmax(-cost, dim=1)
        disparity_values = torch.arange(0, self.max_disparity // 4, 
                                       device=cost.device, dtype=torch.float32)
        disparity_values = disparity_values.view(1, -1, 1, 1)
        
        disparity = torch.sum(prob * disparity_values, dim=1, keepdim=True)
        
        # Upsample to original resolution
        disparity = F.interpolate(disparity, scale_factor=4, mode='bilinear', 
                                 align_corners=False) * 4
        
        return disparity


# ============= Usage Examples =============

def example_traditional_stereo():
    """Traditional stereo matching example"""
    # Camera parameters (example using KITTI dataset parameters)
    focal_length = 718.856  # pixels
    baseline = 0.54  # meters
    img_width = 1242
    img_height = 375
    
    # Create depth estimator
    estimator = StereoDepthEstimator(focal_length, baseline, img_width, img_height)
    
    # Load stereo image pair
    left_img = cv2.imread('left.png')
    right_img = cv2.imread('right.png')
    
    # Compute disparity
    disparity = estimator.compute_disparity_sgbm(left_img, right_img)
    
    # Convert to depth
    depth = estimator.disparity_to_depth(disparity)
    
    # Generate point cloud
    points_3d = estimator.point_cloud_from_depth(depth)
    
    print(f"Disparity range: [{disparity.min():.2f}, {disparity.max():.2f}]")
    print(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}] meters")
    print(f"Point cloud shape: {points_3d.shape}")
    
    return disparity, depth, points_3d


def example_learning_based():
    """Learning-based depth estimation example"""
    # Create model
    model = CostVolumeDepthEstimator(max_disparity=192)
    model.eval()
    
    # Create example input
    left_img = torch.randn(1, 3, 256, 512)
    right_img = torch.randn(1, 3, 256, 512)
    
    # Inference
    with torch.no_grad():
        disparity = model(left_img, right_img)
    
    # Convert to depth
    focal_length = 718.856
    baseline = 0.54
    depth = (baseline * focal_length) / (disparity + 1e-6)
    
    print(f"Output disparity shape: {disparity.shape}")
    print(f"Depth shape: {depth.shape}")
    
    return disparity, depth


if __name__ == "__main__":
    print("=== Traditional Stereo Matching ===")
    # disparity, depth, points_3d = example_traditional_stereo()
    
    print("\n=== Learning-based Method ===")
    disparity, depth = example_learning_based()

'''
Code to read data from tfrecord, mainly the data I generated from Blender
PyTorch version
'''

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob


class TFRecordDataset(Dataset):
    """
    Dataset for reading TFRecord files converted to .npz format
    Data format matches original TF version:
    - z_p, z_c: uint16 depth maps converted to float32 via: value/50000 + 0.3
    - pose_p2c: 4x4 transformation matrix
    """
    def __init__(self, npz_path):
        """
        Args:
            npz_path: Path to .npz file containing z_p, z_c, pose_p2c
        """
        data = np.load(npz_path)
        self.z_p = data['z_p']  # [N, 800, 1200, 1]
        self.z_c = data['z_c']  # [N, 800, 1200, 1]
        self.pose_p2c = data['pose_p2c']  # [N, 4, 4]
        
    def __len__(self):
        return len(self.z_p)
    
    def __getitem__(self, idx):
        return {
            'z_p': torch.from_numpy(self.z_p[idx]).float(),
            'z_c': torch.from_numpy(self.z_c[idx]).float(),
            'pose_p2c': torch.from_numpy(self.pose_p2c[idx]).float()
        }


def read_data(root_path, batchsize, mode='train'):
    """
    Create DataLoader for training/validation/testing
    Equivalent to TF version's read_data function
    
    Args:
        root_path: Root directory containing .npz files
        batchsize: Batch size
        mode: 'train', 'valid', or 'test'
        
    Returns:
        DataLoader that yields batches with z_p, z_c, pose_p2c
    """
    # Find data file based on mode (like TF's get_name_scope logic)
    if mode == 'train':
        path = glob.glob(root_path + 'train*.npz')
    elif mode == 'valid':
        path = glob.glob(root_path + 'valid*.npz')
    else:
        path = glob.glob(root_path + 'test*.npz')
    
    if not path:
        raise FileNotFoundError(f"No {mode} data files found in {root_path}")
    
    # Use first matching file
    dataset = TFRecordDataset(path[0])
    
    # Create DataLoader
    # shuffle and repeat for training, otherwise sequential
    is_train = (mode == 'train')
    dataloader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=is_train,  # equivalent to TF's shuffle(buffer_size=200)
        num_workers=4,     # equivalent to TF's num_parallel_calls=8
        pin_memory=True,   # faster GPU transfer
        drop_last=True     # equivalent to TF's drop_remainder=True
    )
    
    return dataloader


# test code
if __name__ == "__main__":
    train_loader = read_data('./Dataset/', batchsize=1, mode='test')
    for batch in train_loader:
        z_p = batch['z_p']        # [B, 800, 1200, 1]
        z_c = batch['z_c']        # [B, 800, 1200, 1]
        pose_p2c = batch['pose_p2c']  # [B, 4, 4]

        print(f"z_p shape: {z_p.shape}, z_c shape: {z_c.shape}, pose_p2c shape: {pose_p2c.shape}")
        break  # just test one batch
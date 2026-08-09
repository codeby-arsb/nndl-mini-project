import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Adjust path to allow import if src is not in PYTHONPATH
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils import load_rgb_image, preprocess_image

class ColorizationDataset(Dataset):
    def __init__(self, manifest_path: str):
        """
        Args:
            manifest_path: Absolute or relative path to the manifest file (e.g. train.txt)
        """
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
            
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.image_paths = [line.strip() for line in f if line.strip()]
            
        if not self.image_paths:
            raise ValueError(f"Manifest file is empty: {manifest_path}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        rel_path = self.image_paths[idx]
        img_path = os.path.join(self.project_root, rel_path)
        
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        try:
            rgb_image = load_rgb_image(img_path)
            L_np, ab_np = preprocess_image(rgb_image)
            
            # Check shapes
            if L_np.shape != (1, 256, 256) or ab_np.shape != (2, 256, 256):
                raise ValueError(f"Unexpected shape from preprocess_image for {img_path}. L: {L_np.shape}, ab: {ab_np.shape}")
                
            L_tensor = torch.from_numpy(L_np).float()
            ab_tensor = torch.from_numpy(ab_np).float()
            
            return L_tensor, ab_tensor
            
        except Exception as e:
            raise RuntimeError(f"Error processing image {img_path}: {str(e)}")

def create_dataloader(manifest_path: str, batch_size: int, shuffle: bool, num_workers: int = 0, pin_memory: bool = False) -> DataLoader:
    """
    Factory function to create a DataLoader for the ColorizationDataset.
    """
    dataset = ColorizationDataset(manifest_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dataloader

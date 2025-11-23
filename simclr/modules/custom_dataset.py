import numpy as np
import torch
from torch.utils.data import Dataset
from simclr.modules.transformations import TransformsSimCLR


class NPZPairedDataset(Dataset):
    """
    Custom dataset for loading paired images from an npz file.
    
    Expected npz structure:
    - 'X': array of shape (N, 2, 3, H, W) where N is number of samples,
           2 is the pair of images, 3 is RGB channels, H and W are height/width
    - 'PMI': array of shape (N,) containing pointwise mutual information values
    """
    
    def __init__(self, npz_path, transform=None, image_size=32):
        """
        Args:
            npz_path: Path to the npz file
            transform: Optional transform to apply. If None, uses TransformsSimCLR
            image_size: Size of images (default 32 for CIFAR-like images)
        """
        data = np.load(npz_path)
        self.X = data['X']  # Shape: (N, 2, 3, H, W)
        self.PMI = data['PMI']  # Shape: (N,)
        
        # Convert to float32 and normalize to [0, 1] if needed
        if self.X.dtype != np.float32:
            self.X = self.X.astype(np.float32)
        
        # Normalize if values are in [0, 255] range
        if self.X.max() > 1.0:
            self.X = self.X / 255.0
        
        # Convert to torch tensors
        self.X = torch.from_numpy(self.X)
        self.PMI = torch.from_numpy(self.PMI)
        
        # Setup transforms
        if transform is None:
            self.transform = TransformsSimCLR(size=image_size)
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Returns:
            ((x_i, x_j), pmi) where x_i and x_j are the two augmented views
            of the paired images, and pmi is the pointwise mutual information
        """
        # Get the pair of images: shape (2, 3, H, W)
        pair = self.X[idx]  # Shape: (2, 3, H, W)
        
        # Extract individual images
        x_i = pair[0]  # Shape: (3, H, W)
        x_j = pair[1]  # Shape: (3, H, W)
        
        # Convert to PIL Image format (C, H, W) -> (H, W, C) for transforms
        # TransformsSimCLR expects PIL Images, so we need to convert
        # First, permute to (H, W, C) and convert to numpy
        x_i_np = x_i.permute(1, 2, 0).numpy()
        x_j_np = x_j.permute(1, 2, 0).numpy()
        
        # Convert to uint8 if needed for PIL
        if x_i_np.max() <= 1.0:
            x_i_np = (x_i_np * 255).astype(np.uint8)
            x_j_np = (x_j_np * 255).astype(np.uint8)
        else:
            x_i_np = x_i_np.astype(np.uint8)
            x_j_np = x_j_np.astype(np.uint8)
        
        # Convert to PIL Image
        from PIL import Image
        x_i_pil = Image.fromarray(x_i_np)
        x_j_pil = Image.fromarray(x_j_np)
        
        # Apply transforms (SimCLR augmentations)
        # The transform returns two augmented views of the same image
        # For SimCLR, we want two augmented views of each image in the pair
        # So we get two views of x_i and two views of x_j, then return one of each
        x_i_aug1, x_i_aug2 = self.transform(x_i_pil)
        x_j_aug1, x_j_aug2 = self.transform(x_j_pil)
        
        # Return one augmented view of each image
        # This way, x_i_aug1 and x_j_aug1 are the two views that will be compared
        # (they come from different original images, which is what we want for this dataset)
        x_i_aug = x_i_aug1
        x_j_aug = x_j_aug1
        
        # Get PMI value
        pmi = self.PMI[idx].item()
        
        return (x_i_aug, x_j_aug), pmi


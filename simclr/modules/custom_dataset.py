import numpy as np
import torch
from torch.utils.data import Dataset
from simclr.modules.transformations import TransformsSimCLR


class NPZPairDataset(Dataset):
    """
    Custom dataset for loading image pairs from npz file.
    The npz file should contain:
    - 'X': array of shape (N, 2, 3, H, W) - pairs of images
    - 'Noise': array of shape (N, 2, 3, H, W) - latent Z values
    - 'Sigma': covariance matrix
    """
    
    def __init__(self, npz_path, image_size=None, transform=None):
        """
        Args:
            npz_path: Path to the npz file
            image_size: Size to resize images to. If None, uses the original image size from data
            transform: Optional transform to apply. If None, uses TransformsSimCLR
        """
        data = np.load(npz_path)
        self.X = data['X']  # Shape: (N, 2, 3, H, W)
        self.Noise = data['Noise']  # Shape: (N, 2, 3, H, W)
        self.Sigma = data['Sigma']  # Covariance matrix
        
        self.num_samples = self.X.shape[0]
        
        # Auto-detect image size if not provided
        if image_size is None:
            image_size = self.X.shape[-1]  # Use the last dimension (should be H or W, assuming square)
        
        if transform is None:
            self.transform = TransformsSimCLR(size=image_size)
        else:
            self.transform = transform
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns:
            ((x_i, x_j), idx): Tuple of transformed image pairs and index
        """
        # Get the pair of images: shape (2, 3, H, W)
        x_pair = self.X[idx]  # (2, 3, H, W)
        
        # Convert to numpy format expected by transforms (H, W, C)
        x_i = x_pair[0].transpose(1, 2, 0)  # (H, W, 3)
        x_j = x_pair[1].transpose(1, 2, 0)  # (H, W, 3)
        
        # Convert to PIL Image or apply transforms
        # TransformsSimCLR expects PIL Image, so we need to convert
        from PIL import Image
        
        # Normalize to [0, 255] if needed
        if x_i.max() <= 1.0:
            x_i = (x_i * 255).astype(np.uint8)
        else:
            x_i = x_i.astype(np.uint8)
            
        if x_j.max() <= 1.0:
            x_j = (x_j * 255).astype(np.uint8)
        else:
            x_j = x_j.astype(np.uint8)
        
        x_i = Image.fromarray(x_i)
        x_j = Image.fromarray(x_j)
        
        # Apply transforms - apply augmentation to each image in the pair
        if self.transform:
            if hasattr(self.transform, 'train_transform'):
                # Use train_transform directly on each image
                x_i = self.transform.train_transform(x_i)
                x_j = self.transform.train_transform(x_j)
            else:
                # If transform is callable and returns tuple (like TransformsSimCLR)
                # we apply it to get augmented views, but we only need one view per image
                result_i = self.transform(x_i)
                result_j = self.transform(x_j)
                # If transform returns tuple, take first element
                if isinstance(result_i, tuple):
                    x_i = result_i[0]
                else:
                    x_i = result_i
                if isinstance(result_j, tuple):
                    x_j = result_j[0]
                else:
                    x_j = result_j
        
        return (x_i, x_j), idx


import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from simclr import SimCLR
from simclr.modules import get_resnet
from utils import yaml_config_hook


def load_data(npz_path):
    """Load the npz file and return images and PMI values."""
    data = np.load(npz_path)
    X = data['X']  # Shape: (N, 2, 3, H, W)
    PMI = data['PMI']  # Shape: (N,)
    
    # Convert to float32 and normalize
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    if X.max() > 1.0:
        X = X / 255.0
    
    # Ensure values are in [0, 1] range
    X = np.clip(X, 0.0, 1.0)
    
    return X, PMI


def preprocess_image(img_array, image_size=224):
    """
    Preprocess a single image array to match model input format.
    
    Args:
        img_array: Array of shape (3, H, W) with values in [0, 1]
        image_size: Target size for resizing
    
    Returns:
        Preprocessed tensor of shape (3, image_size, image_size)
    """
    # Convert to PIL Image format: (C, H, W) -> (H, W, C)
    img_np = img_array.transpose(1, 2, 0)
    
    # Convert to uint8 if needed
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)
    
    # Convert to PIL Image
    img_pil = Image.fromarray(img_np)
    
    # Apply resize and normalization (matching test_transform)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    
    return transform(img_pil)


def compute_cosine_similarities(model, X, device, batch_size=64, image_size=224):
    """
    Compute cosine similarities between pairs of images using the trained SimCLR model.
    
    Args:
        model: Trained SimCLR model
        X: Array of shape (N, 2, 3, H, W) containing paired images
        device: Device to run computation on
        batch_size: Batch size for processing
        image_size: Target image size for preprocessing
    
    Returns:
        cosine_similarities: Array of shape (N,) containing cosine similarities
    """
    model.eval()
    cosine_similarities = []
    
    N = len(X)
    num_batches = (N + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Computing cosine similarities"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, N)
            batch_X = X[start_idx:end_idx]
            
            # Preprocess images
            x_i_batch = []
            x_j_batch = []
            for j in range(len(batch_X)):
                x_i_processed = preprocess_image(batch_X[j, 0], image_size)
                x_j_processed = preprocess_image(batch_X[j, 1], image_size)
                x_i_batch.append(x_i_processed)
                x_j_batch.append(x_j_processed)
            
            # Stack into batches
            x_i_batch = torch.stack(x_i_batch).to(device)  # Shape: (batch, 3, H, W)
            x_j_batch = torch.stack(x_j_batch).to(device)  # Shape: (batch, 3, H, W)
            
            # Get embeddings (h_i, h_j) from the encoder
            # The model returns (h_i, h_j, z_i, z_j), we want h_i and h_j
            h_i, h_j, _, _ = model(x_i_batch, x_j_batch)
            
            # Compute cosine similarity between h_i and h_j
            # Normalize the embeddings
            h_i_norm = F.normalize(h_i, p=2, dim=1)
            h_j_norm = F.normalize(h_j, p=2, dim=1)
            
            # Compute cosine similarity (dot product of normalized vectors)
            cosine_sim = (h_i_norm * h_j_norm).sum(dim=1).cpu().numpy()
            
            cosine_similarities.extend(cosine_sim)
    
    return np.array(cosine_similarities)


def plot_pmi_vs_cosine_similarity(PMI, cosine_similarities, output_path="pmi_vs_cosine_similarity.png"):
    """
    Create a scatter plot of PMI vs cosine similarity.
    
    Args:
        PMI: Array of PMI values
        cosine_similarities: Array of cosine similarity values
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(PMI, cosine_similarities, alpha=0.5, s=10)
    plt.xlabel('Pointwise Mutual Information (PMI)', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.title('PMI vs Cosine Similarity (SimCLR Embeddings)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(PMI, cosine_similarities)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    try:
        plt.show()
    except:
        # If display is not available (e.g., headless server), just save the file
        print("Display not available, plot saved to file only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PMI vs Cosine Similarity")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    
    # Additional arguments for this script (only if not already in config)
    if "output_plot" not in config:
        parser.add_argument("--output_plot", type=str, 
                           default="pmi_vs_cosine_similarity.png",
                           help="Output path for the plot")
    if "batch_size" not in config:
        parser.add_argument("--batch_size", type=int, default=64,
                           help="Batch size for computing similarities")
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {args.device}")
    print(f"Loading data from: {args.npz_path}")
    
    # Load data
    X, PMI = load_data(args.npz_path)
    print(f"Loaded {len(X)} samples")
    print(f"X shape: {X.shape}, PMI shape: {PMI.shape}")
    
    # Load trained model
    print(f"Loading model from: {args.model_path}")
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features
    
    model = SimCLR(encoder, args.projection_dim, n_features)
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
    
    if not os.path.exists(model_fp):
        raise FileNotFoundError(f"Model checkpoint not found: {model_fp}")
    
    model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)
    model.eval()
    
    print("Model loaded successfully")
    
    # Compute cosine similarities
    print("Computing cosine similarities...")
    cosine_similarities = compute_cosine_similarities(
        model, X, args.device, batch_size=args.batch_size, image_size=args.image_size
    )
    
    print(f"Computed {len(cosine_similarities)} cosine similarities")
    print(f"Cosine similarity range: [{cosine_similarities.min():.4f}, {cosine_similarities.max():.4f}]")
    print(f"PMI range: [{PMI.min():.4f}, {PMI.max():.4f}]")
    
    # Create plot
    print("Creating plot...")
    plot_pmi_vs_cosine_similarity(PMI, cosine_similarities, args.output_plot)
    
    print("Done!")


import os
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader

# SimCLR
from simclr import SimCLR
from simclr.modules import get_resnet
from simclr.modules.custom_dataset import NPZPairDataset
from simclr.modules.transformations import TransformsSimCLR

from utils import yaml_config_hook


def calculate_pmi_batch(noise_data, sigma):
    """
    Calculate Pointwise Mutual Information (PMI) for all pairs of noise vectors.
    Optimized to pre-compute covariance submatrices.
    
    Args:
        noise_data: Array of shape (N, 2, 3, 32, 32) - all noise pairs
        sigma: Full covariance matrix
    
    Returns:
        PMI values: Array of shape (N,)
    """
    N = noise_data.shape[0]
    dim = 3 * 32 * 32  # 3072
    
    # Extract submatrices once (these are shared across all samples)
    Sigma_11 = sigma[:dim, :dim]
    Sigma_22 = sigma[dim:, dim:]
    Sigma_joint = sigma
    
    # Pre-compute determinants (these are also shared)
    det_11 = None
    det_22 = None
    det_joint = None
    try:
        det_11 = np.linalg.det(Sigma_11)
        det_22 = np.linalg.det(Sigma_22)
        det_joint = np.linalg.det(Sigma_joint)
    except:
        pass
    
    # Mean vectors (assuming zero mean)
    mean1 = np.zeros(dim)
    mean2 = np.zeros(dim)
    mean_joint = np.zeros(2 * dim)
    
    # Create distributions once
    try:
        joint_dist = multivariate_normal(mean=mean_joint, cov=Sigma_joint, allow_singular=True)
        marginal1 = multivariate_normal(mean=mean1, cov=Sigma_11, allow_singular=True)
        marginal2 = multivariate_normal(mean=mean2, cov=Sigma_22, allow_singular=True)
    except Exception as e:
        print(f"Warning: Error creating distributions: {e}")
        # Fallback to analytical if possible
        if det_11 is not None and det_22 is not None and det_joint is not None:
            if det_11 > 0 and det_22 > 0 and det_joint > 0:
                mi_constant = 0.5 * np.log(det_11 * det_22 / det_joint)
                return np.full(N, mi_constant)
        return np.zeros(N)
    
    pmi_values = []
    
    for idx in range(N):
        noise_pair = noise_data[idx]  # (2, 3, 32, 32)
        
        # Flatten the noise vectors
        z1 = noise_pair[0].flatten()  # (3072,)
        z2 = noise_pair[1].flatten()  # (3072,)
        
        try:
            # Compute log probabilities
            z_joint = np.concatenate([z1, z2])
            log_p_joint = joint_dist.logpdf(z_joint)
            log_p1 = marginal1.logpdf(z1)
            log_p2 = marginal2.logpdf(z2)
            
            # PMI = log(p(z1, z2) / (p(z1) * p(z2))) = log(p(z1, z2)) - log(p(z1)) - log(p(z2))
            pmi = log_p_joint - log_p1 - log_p2
            
            # Handle invalid values
            if not np.isfinite(pmi):
                pmi = 0.0
                
        except Exception as e:
            # Fallback: use analytical formula if available
            if det_11 is not None and det_22 is not None and det_joint is not None:
                if det_11 > 0 and det_22 > 0 and det_joint > 0:
                    pmi = 0.5 * np.log(det_11 * det_22 / det_joint)
                else:
                    pmi = 0.0
            else:
                pmi = 0.0
        
        pmi_values.append(pmi)
    
    return np.array(pmi_values)


def extract_cosine_similarities(model, dataset, device, batch_size=32):
    """
    Extract cosine similarities for all pairs in the dataset.
    
    Returns:
        cosine_similarities: Array of shape (N,) with cosine similarity for each pair
    """
    model.eval()
    cosine_similarities = []
    
    # Create a simple transform without augmentation for evaluation
    # Use the same size as the dataset was trained with
    image_size = dataset.X.shape[-1]  # Get size from data (should be 32)
    eval_transform = TransformsSimCLR(size=image_size).test_transform
    
    # Process in batches for efficiency
    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch_x_i = []
            batch_x_j = []
            
            for idx in range(batch_start, batch_end):
                # Get the original pair
                x_pair = dataset.X[idx]  # (2, 3, H, W)
                
                # Convert to tensor format
                x_i = x_pair[0].transpose(1, 2, 0)  # (H, W, 3)
                x_j = x_pair[1].transpose(1, 2, 0)  # (H, W, 3)
                
                # Normalize if needed
                if x_i.max() <= 1.0:
                    x_i = (x_i * 255).astype(np.uint8)
                    x_j = (x_j * 255).astype(np.uint8)
                else:
                    x_i = x_i.astype(np.uint8)
                    x_j = x_j.astype(np.uint8)
                
                from PIL import Image
                x_i = Image.fromarray(x_i)
                x_j = Image.fromarray(x_j)
                
                # Apply test transform (no augmentation)
                x_i = eval_transform(x_i)
                x_j = eval_transform(x_j)
                
                batch_x_i.append(x_i)
                batch_x_j.append(x_j)
            
            # Stack into batch
            batch_x_i = torch.stack(batch_x_i).to(device)
            batch_x_j = torch.stack(batch_x_j).to(device)
            
            # Get embeddings
            h_i, h_j, z_i, z_j = model(batch_x_i, batch_x_j)
            
            # Calculate cosine similarity on the projection head output (z)
            z_i_norm = F.normalize(z_i, p=2, dim=1)
            z_j_norm = F.normalize(z_j, p=2, dim=1)
            cos_sim = (z_i_norm * z_j_norm).sum(dim=1).cpu().numpy()
            
            cosine_similarities.extend(cos_sim)
            
            if batch_end % 100 == 0 or batch_end == len(dataset):
                print(f"Processed {batch_end}/{len(dataset)} samples")
    
    return np.array(cosine_similarities)


def main():
    parser = argparse.ArgumentParser(description="Analyze PMI vs Cosine Similarity for custom dataset")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    
    # Add custom arguments
    parser.add_argument("--npz_path", type=str, 
                       default="datasets/datasets_4624753/simple_fig1_DAG_rhoTheta_0p010/simple_fig1_DAG_0.npz",
                       help="Path to the npz file")
    parser.add_argument("--model_path", type=str, default="save",
                       help="Path to directory containing model checkpoints")
    parser.add_argument("--epoch_num", type=str, default=None,
                       help="Epoch number to load checkpoint from, or 'best' for best checkpoint (default: auto-detect latest)")
    parser.add_argument("--output_plot", type=str, default="pmi_vs_cosine_sim.png",
                       help="Output path for the scatter plot")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for extracting cosine similarities")
    
    args = parser.parse_args()
    
    # Setup device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Determine which checkpoint to load
    if args.epoch_num is None or args.epoch_num == "latest":
        # Find the latest checkpoint
        checkpoint_files = []
        if os.path.exists(args.model_path):
            for f in os.listdir(args.model_path):
                if f.startswith("checkpoint_") and f.endswith(".tar") and f != "checkpoint_best.tar":
                    try:
                        epoch = int(f.split("_")[1].split(".")[0])
                        checkpoint_files.append((epoch, f))
                    except:
                        pass
        
        if checkpoint_files:
            args.epoch_num = max(epoch for epoch, _ in checkpoint_files)
            print(f"Auto-detected latest checkpoint: epoch {args.epoch_num}")
        else:
            raise ValueError(f"No checkpoint files found in {args.model_path}")
    elif args.epoch_num == "best":
        # Use best checkpoint
        args.epoch_num = "best"
        print("Using best checkpoint")
    else:
        # Use specified epoch number
        try:
            args.epoch_num = int(args.epoch_num)
        except ValueError:
            raise ValueError(f"Invalid epoch_num: {args.epoch_num}. Must be an integer, 'best', or 'latest'")
    
    # Load dataset
    print(f"Loading dataset from {args.npz_path}")
    image_size = None if args.image_size == 224 else args.image_size
    dataset = NPZPairDataset(args.npz_path, image_size=image_size)
    actual_size = dataset.X.shape[-1]
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Image size: {actual_size}x{actual_size}")
    
    # Initialize and load model
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features
    model = SimCLR(encoder, args.projection_dim, n_features)
    
    if args.epoch_num == "best":
        model_fp = os.path.join(args.model_path, "checkpoint_best.tar")
    else:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
    
    if not os.path.exists(model_fp):
        raise FileNotFoundError(f"Model file not found: {model_fp}")
    
    print(f"Loading model from {model_fp}")
    model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)
    model.eval()
    
    # Extract cosine similarities
    print("\nExtracting cosine similarities...")
    cosine_similarities = extract_cosine_similarities(model, dataset, args.device, batch_size=args.batch_size)
    print(f"Extracted {len(cosine_similarities)} cosine similarities")
    
    # Calculate PMI for all pairs
    print("\nCalculating PMI for all pairs...")
    pmi_values = calculate_pmi_batch(dataset.Noise, dataset.Sigma)
    print(f"Calculated {len(pmi_values)} PMI values")
    
    # Create scatter plot
    print("\nCreating scatter plot...")
    plt.figure(figsize=(10, 8))
    plt.scatter(pmi_values, cosine_similarities, alpha=0.5, s=10)
    plt.xlabel('Pointwise Mutual Information (PMI)', fontsize=12)
    plt.ylabel('Cosine Similarity (SimCLR)', fontsize=12)
    plt.title('PMI vs Cosine Similarity', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Calculate and display correlation
    correlation = np.corrcoef(pmi_values, cosine_similarities)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {args.output_plot}")
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"PMI - Mean: {pmi_values.mean():.4f}, Std: {pmi_values.std():.4f}")
    print(f"PMI - Min: {pmi_values.min():.4f}, Max: {pmi_values.max():.4f}")
    print(f"Cosine Similarity - Mean: {cosine_similarities.mean():.4f}, Std: {cosine_similarities.std():.4f}")
    print(f"Cosine Similarity - Min: {cosine_similarities.min():.4f}, Max: {cosine_similarities.max():.4f}")
    print(f"Correlation: {correlation:.4f}")


if __name__ == "__main__":
    main()


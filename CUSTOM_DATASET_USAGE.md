# Custom Dataset Usage Guide

This guide explains how to use SimCLR with your custom npz dataset to generate cosine similarities and compare them with PMI values.

## Overview

The workflow is split into two scripts:

1. **`train_simclr_custom.py`**: Trains SimCLR on your custom dataset and saves the model
2. **`analyze_pmi_cosine.py`**: Loads a trained model, extracts cosine similarities, calculates PMI, and creates the scatter plot

This separation allows you to:
- Train once and analyze multiple times
- Try different analysis parameters without retraining
- Share trained models with others

## Dataset Format

Your npz file should contain:
- **X**: Array of shape `(N, 2, 3, H, W)` - pairs of images
  - N: number of samples (e.g., 10,000)
  - 2: pair of images
  - 3: RGB channels
  - H, W: image height and width (e.g., 32x32)
  
- **Noise**: Array of shape `(N, 2, 3, H, W)` - latent Z values used to generate X
  
- **Sigma**: Covariance matrix with block structure:
  ```
  Sigma = [[Sigma_11, Sigma_12],
           [Sigma_21, Sigma_22]]
  ```
  where each block is (3*H*W) x (3*H*W)

## Usage

### Step 1: Train SimCLR

Train SimCLR on your custom dataset:

```bash
python train_simclr_custom.py --npz_path datasets/datasets_4624753/simple_fig1_DAG_rhoTheta_0p010/simple_fig1_DAG_0.npz
```

With custom parameters:

```bash
python train_simclr_custom.py \
    --npz_path datasets/datasets_4624753/simple_fig1_DAG_rhoTheta_0p010/simple_fig1_DAG_0.npz \
    --image_size 32 \
    --batch_size 128 \
    --epochs 100 \
    --resnet resnet18 \
    --model_path save
```

### Step 2: Analyze PMI vs Cosine Similarity

After training, run the analysis. You can use the best checkpoint:

```bash
python analyze_pmi_cosine.py \
    --npz_path datasets/datasets_4624753/simple_fig1_DAG_rhoTheta_0p010/simple_fig1_DAG_0.npz \
    --model_path save \
    --epoch_num best \
    --output_plot pmi_vs_cosine_sim.png
```

Or use a specific epoch:

```bash
python analyze_pmi_cosine.py \
    --npz_path datasets/datasets_4624753/simple_fig1_DAG_rhoTheta_0p010/simple_fig1_DAG_0.npz \
    --model_path save \
    --epoch_num 100 \
    --output_plot pmi_vs_cosine_sim.png
```

The script will auto-detect the latest checkpoint if `--epoch_num` is not specified:

```bash
python analyze_pmi_cosine.py \
    --npz_path datasets/datasets_4624753/simple_fig1_DAG_rhoTheta_0p010/simple_fig1_DAG_0.npz \
    --model_path save \
    --output_plot my_analysis.png
```

## Parameters

### train_simclr_custom.py

- `--npz_path`: Path to your npz file (required)
- `--image_size`: Image size for resizing (default: auto-detected from data, typically 32)
- `--batch_size`: Training batch size (default: from config.yaml)
- `--epochs`: Number of training epochs (default: from config.yaml)
- `--resnet`: ResNet architecture - "resnet18" or "resnet50" (default: from config.yaml)
- `--model_path`: Path to save model checkpoints (default: "save")
- `--reload`: Load existing checkpoint if available (uses `--epoch_num`)
- `--epoch_num`: Epoch number to load checkpoint from (default: from config.yaml)

### analyze_pmi_cosine.py

- `--npz_path`: Path to your npz file (required)
- `--model_path`: Path to saved model checkpoints (default: "save")
- `--epoch_num`: Epoch number to load checkpoint from, or `"best"` for best checkpoint, or `"latest"` for latest (default: auto-detects latest)
- `--output_plot`: Output filename for the scatter plot (default: "pmi_vs_cosine_sim.png")
- `--batch_size`: Batch size for extracting cosine similarities (default: 32)
- `--resnet`: ResNet architecture - must match the trained model (default: from config.yaml)
- `--projection_dim`: Projection dimension - must match the trained model (default: from config.yaml)

## Output

### train_simclr_custom.py

- Saves model checkpoints in the `model_path` directory
- **Best checkpoint**: Automatically saved as `checkpoint_best.tar` whenever training loss improves
- **Periodic checkpoints**: Saved every 10 epochs as `checkpoint_{epoch_num}.tar`
- **Final checkpoint**: Saved at the end of training
- TensorBoard logs are saved to `runs/` directory
- Training prints `*New best!*` when a new best checkpoint is saved

### analyze_pmi_cosine.py

- Generates a scatter plot saved as `output_plot` (default: `pmi_vs_cosine_sim.png`)
- Prints statistics including:
  - PMI statistics (mean, std, min, max)
  - Cosine similarity statistics
  - Correlation coefficient between PMI and cosine similarity

## Notes

- Both scripts automatically detect the image size from your data
- For 32x32 images (CIFAR-10 style), ResNet18 is recommended
- Training can take a while depending on dataset size and epochs
- The PMI calculation uses multivariate Gaussian distributions
- Cosine similarities are computed from the projection head outputs (z_i, z_j)
- Make sure `--resnet` and `--projection_dim` match between training and analysis scripts

## Troubleshooting

**Out of memory errors**: Reduce `--batch_size` in the training script

**Slow PMI calculation**: The PMI calculation is optimized but may still take time for large datasets

**Model not loading**: 
- Make sure `--model_path` and `--epoch_num` point to an existing checkpoint file
- If `--epoch_num` is not specified, the analysis script will auto-detect the latest checkpoint
- Ensure `--resnet` and `--projection_dim` match the training configuration

**Mismatched model architecture**: The analysis script must use the same `--resnet` and `--projection_dim` as the training script


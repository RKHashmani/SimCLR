import os
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.custom_dataset import NPZPairDataset

from model import load_optimizer, save_model
from utils import yaml_config_hook


def train(args, train_loader, model, criterion, optimizer, writer):
    """Training loop for SimCLR"""
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True) if torch.cuda.is_available() else x_i
        x_j = x_j.cuda(non_blocking=True) if torch.cuda.is_available() else x_j

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()

        if args.nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0 and writer:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def main():
    parser = argparse.ArgumentParser(description="Train SimCLR on custom dataset")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    
    # Add custom arguments
    parser.add_argument("--npz_path", type=str, 
                       default="datasets/datasets_4624753/simple_fig1_DAG_rhoTheta_0p010/simple_fig1_DAG_0.npz",
                       help="Path to the npz file")
    
    args = parser.parse_args()
    
    # Setup device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes
    args.global_step = 0
    args.current_epoch = 0
    
    # Create output directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Load dataset
    print(f"Loading dataset from {args.npz_path}")
    # Auto-detect image size from data (default from config is 224, but data is likely 32x32)
    # Pass None to auto-detect, or use args.image_size if explicitly set to non-default
    image_size = None if args.image_size == 224 else args.image_size
    dataset = NPZPairDataset(args.npz_path, image_size=image_size)
    actual_size = dataset.X.shape[-1]  # Assuming square images
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Image size: {actual_size}x{actual_size}")
    
    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    
    # Initialize model
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features
    model = SimCLR(encoder, args.projection_dim, n_features)
    
    # Load model if reload is enabled
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
        if os.path.exists(model_fp):
            print(f"Loading model from {model_fp}")
            model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
        else:
            print(f"Model file not found: {model_fp}")
            print("Starting training from scratch...")
    
    model = model.to(args.device)
    
    # Training
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter() if args.nr == 0 else None
    
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)
    
    # Track best loss for saving best checkpoint
    best_loss = float('inf')
    best_epoch = -1
    
    print("Starting training...")
    for epoch in range(args.start_epoch, args.epochs):
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)
        avg_loss = loss_epoch / len(train_loader)
        
        if scheduler:
            scheduler.step()
        
        if args.nr == 0:
            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            
            # Save best checkpoint if loss improved
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch
                best_checkpoint_path = os.path.join(args.model_path, "checkpoint_best.tar")
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), best_checkpoint_path)
                else:
                    torch.save(model.state_dict(), best_checkpoint_path)
                print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {avg_loss:.6f}\t lr: {round(lr, 5)}\t *New best!*")
            else:
                print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {avg_loss:.6f}\t lr: {round(lr, 5)}")
            
            # Save periodic checkpoints every 10 epochs
            if epoch % 10 == 0:
                save_model(args, model, optimizer)
            
            args.current_epoch += 1
    
    # Save final model
    save_model(args, model, optimizer)
    if writer:
        writer.close()
    
    print(f"\nTraining complete! Model saved to {args.model_path}")
    print(f"Final checkpoint: checkpoint_{args.current_epoch}.tar")
    print(f"Best checkpoint: checkpoint_best.tar (epoch {best_epoch}, loss: {best_loss:.6f})")


if __name__ == "__main__":
    main()


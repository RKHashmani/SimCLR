import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from simclr import SimCLR
from simclr.modules import get_resnet
from utils import NPZPairDataset


def load_simclr_model(
    checkpoint_path: str, resnet: str, projection_dim: int, device: torch.device
) -> SimCLR:
    encoder = get_resnet(resnet, pretrained=False)
    n_features = encoder.fc.in_features
    model = SimCLR(encoder, projection_dim, n_features)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device)


def compute_similarities(
    model: SimCLR, loader: DataLoader, device: torch.device, use_projector: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_cos = []
    all_pmi = []
    for (x_i, x_j), pmi in loader:
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        pmi = pmi.to(device)

        with torch.no_grad():
            h_i = model.encoder(x_i)
            h_j = model.encoder(x_j)
            feats_i = model.projector(h_i) if use_projector else h_i
            feats_j = model.projector(h_j) if use_projector else h_j

            feats_i = F.normalize(feats_i, dim=1)
            feats_j = F.normalize(feats_j, dim=1)
            cos_sim = torch.sum(feats_i * feats_j, dim=1)

        all_cos.append(cos_sim.cpu())
        all_pmi.append(pmi.cpu())

    return torch.cat(all_pmi), torch.cat(all_cos)


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if not os.path.exists(args.npz_path):
        raise FileNotFoundError(f"npz file not found: {args.npz_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint_path}")

    base_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.image_size, args.image_size)),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = NPZPairDataset(args.npz_path, transform=base_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    model = load_simclr_model(args.checkpoint_path, args.resnet, args.projection_dim, device)
    pmi, cos_sim = compute_similarities(model, loader, device, use_projector=not args.use_encoder_only)

    plt.figure(figsize=(8, 6))
    plt.scatter(pmi.numpy(), cos_sim.numpy(), s=8, alpha=0.5)
    plt.xlabel("Pointwise Mutual Information (PMI)")
    plt.ylabel("Cosine similarity")
    plt.title("PMI vs Cosine Similarity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=200)
    print(f"Saved plot to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PMI vs cosine similarity for a trained SimCLR model.")
    parser.add_argument("--npz_path", required=True, help="Path to the paired-image npz file.")
    parser.add_argument("--checkpoint_path", required=True, help="Checkpoint produced by training (checkpoint_*.tar).")
    parser.add_argument("--output_path", default="pmi_vs_similarity.png", help="Where to save the scatter plot.")
    parser.add_argument("--resnet", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=32, help="Images are 32x32 in the provided npz.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument(
        "--use_encoder_only",
        action="store_true",
        help="Use encoder outputs instead of the projection head for cosine similarity.",
    )

    main(parser.parse_args())

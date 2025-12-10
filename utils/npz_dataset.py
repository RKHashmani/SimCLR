import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple


class NPZPairDataset(Dataset):
    """
    Dataset for paired images stored in a NPZ file with layout:
        X: (N, 2, C, H, W) containing positive pairs
        PMI: (N,) containing pointwise mutual information values
    Augmentations are intentionally disabled; optional tensor-level transforms
    can be provided for normalization/resizing.
    """

    def __init__(
        self,
        npz_path: str,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        scale_to_unit: bool = True,
    ) -> None:
        with np.load(npz_path, allow_pickle=False) as data:
            if "X" not in data:
                raise ValueError("Expected key 'X' in npz file")
            self.images = np.array(data["X"])
            self.pmi = np.array(data["PMI"]) if "PMI" in data else None

        if self.images.ndim != 5 or self.images.shape[1] != 2:
            raise ValueError(
                f"Expected images of shape (N, 2, C, H, W), got {self.images.shape}"
            )

        self.transform = transform
        self.scale_to_unit = scale_to_unit
        # Heuristic: if stored as bytes/ints in [0, 255], scale to [0, 1].
        self._scale_factor = 255.0 if self.scale_to_unit and self.images.max() > 1.0 else 1.0

    def __len__(self) -> int:
        return self.images.shape[0]

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(arr).float() / self._scale_factor
        if self.transform:
            tensor = self.transform(tensor)
        return tensor

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], float]:
        x_i = self._to_tensor(self.images[idx, 0])
        x_j = self._to_tensor(self.images[idx, 1])
        label = float(self.pmi[idx]) if self.pmi is not None else float(idx)
        return (x_i, x_j), label

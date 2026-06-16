PyTorch Dataset wrapping the synthetic I/Q generator and SigMF file loader.

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

from .signal_generator import (
    SignalDatasetGenerator, GeneratorConfig,
    SIGNAL_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
)


class SyntheticIQDataset(Dataset):
  
    On-the-fly synthetic I/Q dataset. Generates one example per __getitem__ call.

    Args:
        signal_classes: List of signal class names to include
        n_samples_per_class: Number of examples per class
        n_iq: I/Q samples per example
        snr_range: (min_dB, max_dB) for random SNR per example
        seed: RNG seed (set for reproducibility)
        augment: Apply additional augmentation (time shift, amplitude scale)
    """

    def __init__(
        self,
        signal_classes: List[str],
        n_samples_per_class: int = 1000,
        n_iq: int = 1024,
        snr_range: Tuple[float, float] = (-10.0, 30.0),
        seed: Optional[int] = None,
        augment: bool = False,
    ):
        self.signal_classes = signal_classes
        self.n_samples_per_class = n_samples_per_class
        self.snr_range = snr_range
        self.augment = augment

        self.cfg = GeneratorConfig(n_samples=n_iq, seed=seed)
        self.gen = SignalDatasetGenerator(self.cfg)

        # Pre-generate all examples (cached in memory for speed)
        total = len(signal_classes) * n_samples_per_class
        print(f"Generating {total} I/Q examples ({len(signal_classes)} classes × {n_samples_per_class})...")
        self.X, self.y = self.gen.generate_batch(
            signal_classes, n_samples_per_class, snr_range
        )
        print(f"Dataset ready: X={self.X.shape}, y={self.y.shape}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])   # [2, N]
        y = torch.tensor(self.y[idx], dtype=torch.long)

        if self.augment:
            x = self._augment(x)

        return x, y

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Random amplitude scaling and phase rotation."""
        # Amplitude scale
        scale = torch.empty(1).uniform_(0.7, 1.0)
        x = x * scale
        # Phase rotation (same for I and Q via complex multiplication)
        angle = torch.empty(1).uniform_(0, 2 * 3.14159)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        i_rot = x[0] * cos_a - x[1] * sin_a
        q_rot = x[0] * sin_a + x[1] * cos_a
        return torch.stack([i_rot, q_rot])

    @property
    def n_classes(self) -> int:
        return len(self.signal_classes)

    @property
    def class_names(self) -> List[str]:
        return self.signal_classes


class SigMFDataset(Dataset):
    """
    Load I/Q captures stored in SigMF format (.sigmf-data + .sigmf-meta).

    Supports labeled captures (label in meta annotations) and unlabeled
    captures (returns filename as identifier).
    """

    def __init__(
        self,
        data_dir: str,
        n_iq: int = 1024,
        stride: int = 512,
        signal_classes: Optional[List[str]] = None,
    ):
        self.n_iq = n_iq
        self.stride = stride
        self.signal_classes = signal_classes or SIGNAL_CLASSES

        self._segments: List[Tuple[np.ndarray, int]] = []
        data_path = Path(data_dir)

        for meta_file in sorted(data_path.glob("*.sigmf-meta")):
            data_file = meta_file.with_suffix(".sigmf-data")
            if not data_file.exists():
                continue
            self._load_sigmf(meta_file, data_file)

        print(f"SigMFDataset: loaded {len(self._segments)} segments from {data_dir}")

    def _load_sigmf(self, meta_file: Path, data_file: Path):
        with open(meta_file) as f:
            meta = json.load(f)

        # Read raw I/Q
        dtype_str = meta.get("global", {}).get("core:datatype", "cf32_le")
        if "cf32" in dtype_str:
            raw = np.fromfile(data_file, dtype=np.complex64)
        elif "ci16" in dtype_str:
            raw = np.fromfile(data_file, dtype=np.int16).astype(np.float32)
            raw = (raw[0::2] + 1j * raw[1::2]) / 32768.0
        else:
            raw = np.fromfile(data_file, dtype=np.complex64)

        # Determine label from annotations
        label = -1
        annotations = meta.get("annotations", [])
        if annotations:
            sig_name = annotations[0].get("core:label", "").lower()
            for cls in self.signal_classes:
                if cls.replace("_", "") in sig_name.replace("_", ""):
                    label = CLASS_TO_IDX.get(cls, -1)
                    break

        # Slice into fixed-length segments with stride
        i = 0
        while i + self.n_iq <= len(raw):
            seg = raw[i:i + self.n_iq]
            seg_norm = seg / (np.abs(seg).max() + 1e-12)
            self._segments.append((seg_norm, label))
            i += self.stride

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        iq, label = self._segments[idx]
        x = torch.stack([
            torch.from_numpy(iq.real.astype(np.float32)),
            torch.from_numpy(iq.imag.astype(np.float32)),
        ])
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def make_dataloaders(
    signal_classes: List[str],
    n_train: int = 5000,
    n_val: int = 1000,
    n_test: int = 1000,
    n_iq: int = 1024,
    snr_range: Tuple[float, float] = (-10.0, 30.0),
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience factory for train/val/test DataLoaders."""

    train_ds = SyntheticIQDataset(signal_classes, n_train, n_iq, snr_range, seed=seed, augment=True)
    val_ds   = SyntheticIQDataset(signal_classes, n_val,   n_iq, snr_range, seed=seed+1)
    test_ds  = SyntheticIQDataset(signal_classes, n_test,  n_iq, snr_range, seed=seed+2)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    return train_loader, val_loader, test_loader

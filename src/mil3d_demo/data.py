from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SyntheticBagConfig:
    n_patients: int = 160
    positive_fraction: float = 0.5
    min_blocks: int = 5
    max_blocks: int = 10
    depth: int = 16
    height: int = 32
    width: int = 32
    seed: int = 7


class SyntheticCTBagDataset(Dataset):
    """Synthetic bag-level dataset for privacy-safe MIL experiments.

    Positive bags contain one or two small Gaussian blobs in randomly selected
    3D blocks. Negative bags contain only synthetic lung-like background noise.
    """

    def __init__(self, config: SyntheticBagConfig, indices: Sequence[int] | None = None):
        self.config = config
        self._bags, self._labels, self._metadata = self._generate(config)
        if indices is None:
            self.indices = list(range(len(self._labels)))
        else:
            self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        idx = self.indices[item]
        bag = torch.from_numpy(self._bags[idx]).float()
        label = torch.tensor(self._labels[idx], dtype=torch.float32)
        return bag, label, self._metadata[idx]

    @staticmethod
    def _make_lung_mask(depth: int, height: int, width: int) -> np.ndarray:
        z, y, x = np.meshgrid(
            np.linspace(-1.0, 1.0, depth),
            np.linspace(-1.0, 1.0, height),
            np.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        left = ((x + 0.35) / 0.42) ** 2 + (y / 0.72) ** 2 + (z / 0.92) ** 2 < 1.0
        right = ((x - 0.35) / 0.42) ** 2 + (y / 0.72) ** 2 + (z / 0.92) ** 2 < 1.0
        return (left | right).astype(np.float32)

    @staticmethod
    def _add_blob(block: np.ndarray, rng: np.random.Generator, mask: np.ndarray) -> None:
        depth, height, width = mask.shape
        valid = np.argwhere(mask > 0)
        center_z, center_y, center_x = valid[rng.integers(0, len(valid))]
        z, y, x = np.meshgrid(
            np.arange(depth),
            np.arange(height),
            np.arange(width),
            indexing="ij",
        )
        sigma = rng.uniform(1.5, 2.8)
        blob = np.exp(
            -(
                (z - center_z) ** 2
                + (y - center_y) ** 2
                + (x - center_x) ** 2
            )
            / (2.0 * sigma**2)
        )
        block += (0.65 * blob * mask).astype(np.float32)

    @classmethod
    def _generate(
        cls, config: SyntheticBagConfig
    ) -> Tuple[List[np.ndarray], List[int], List[Dict[str, int]]]:
        rng = np.random.default_rng(config.seed)
        mask = cls._make_lung_mask(config.depth, config.height, config.width)
        n_pos = int(round(config.n_patients * config.positive_fraction))
        labels = np.array([1] * n_pos + [0] * (config.n_patients - n_pos))
        rng.shuffle(labels)

        bags: List[np.ndarray] = []
        metadata: List[Dict[str, int]] = []
        for patient_idx, label in enumerate(labels.tolist()):
            n_blocks = int(rng.integers(config.min_blocks, config.max_blocks + 1))
            bag = np.zeros((n_blocks, 1, config.depth, config.height, config.width), dtype=np.float32)
            lesion_blocks: Iterable[int] = []
            if label == 1:
                n_lesion_blocks = int(rng.integers(1, min(3, n_blocks) + 1))
                lesion_blocks = rng.choice(n_blocks, size=n_lesion_blocks, replace=False).tolist()

            for block_idx in range(n_blocks):
                block = rng.normal(0.18, 0.035, size=mask.shape).astype(np.float32) * mask
                texture = rng.normal(0.0, 0.025, size=mask.shape).astype(np.float32) * mask
                block += texture
                if block_idx in lesion_blocks:
                    cls._add_blob(block, rng, mask)
                bag[block_idx, 0] = np.clip(block, 0.0, 1.0)

            bags.append(bag)
            metadata.append(
                {
                    "patient_index": patient_idx,
                    "n_blocks": n_blocks,
                    "label": int(label),
                }
            )

        return bags, labels.astype(int).tolist(), metadata


def stratified_indices(labels: Sequence[int], test_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for cls in sorted(set(labels_array.tolist())):
        cls_idx = np.flatnonzero(labels_array == cls)
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(len(cls_idx) * test_fraction)))
        test_idx.extend(cls_idx[:n_test].tolist())
        train_idx.extend(cls_idx[n_test:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def collate_bags(batch):
    bags, labels, metadata = zip(*batch)
    return list(bags), torch.stack(labels), list(metadata)


def make_dataloaders(
    config: SyntheticBagConfig,
    batch_size: int = 8,
    test_fraction: float = 0.25,
) -> Tuple[DataLoader, DataLoader]:
    full = SyntheticCTBagDataset(config)
    train_idx, test_idx = stratified_indices(full._labels, test_fraction, config.seed + 100)
    train_ds = SyntheticCTBagDataset(config, train_idx)
    test_ds = SyntheticCTBagDataset(config, test_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_bags)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_bags)
    return train_loader, test_loader

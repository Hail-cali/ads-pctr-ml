"""Criteo Display Ads dataset preprocessor and DataLoader.

Supports reczoo/Criteo_x1 pre-encoded format:
  - Label: 0/1 (CTR ~3.4%)
  - I1~I13: Float (dense, pre-normalized) -> standardize
  - C1~C26: Integer indices (sparse, pre-encoded) -> modulo bucket

For large datasets (45M+ rows), uses chunked reading + numpy memmap
to avoid loading everything into RAM at once.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

LABEL_COL = "label"
DENSE_COLS = [f"I{i}" for i in range(1, 14)]
SPARSE_COLS = [f"C{i}" for i in range(1, 27)]
ALL_COLS = [LABEL_COL] + DENSE_COLS + SPARSE_COLS
NUM_DENSE = 13
NUM_SPARSE = 26
HASH_BUCKET_SIZE = 100_000


class CriteoDataset(Dataset):
    def __init__(self, dense: np.ndarray, sparse: np.ndarray, labels: np.ndarray):
        self.dense = torch.FloatTensor(dense)
        self.sparse = torch.LongTensor(sparse)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.dense[idx], self.sparse[idx], self.labels[idx]


class CriteoMemmapDataset(Dataset):
    """Memory-mapped dataset for large-scale training."""

    def __init__(self, dense: np.memmap, sparse: np.memmap, labels: np.memmap):
        self.dense = dense
        self.sparse = sparse
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.dense[idx].copy()),
            torch.LongTensor(self.sparse[idx].copy()),
            torch.FloatTensor([self.labels[idx]])[0],
        )


class CriteoPreprocessor:
    def __init__(self, hash_bucket_size: int = HASH_BUCKET_SIZE, sample_size: Optional[int] = None):
        self.hash_bucket_size = hash_bucket_size
        self.sample_size = sample_size
        self.dense_mean: Optional[np.ndarray] = None
        self.dense_std: Optional[np.ndarray] = None

    def _encode_sparse_array(self, sparse_raw: np.ndarray) -> np.ndarray:
        result = np.zeros_like(sparse_raw, dtype=np.int64)
        for i in range(NUM_SPARSE):
            result[:, i] = i * self.hash_bucket_size + (sparse_raw[:, i] % self.hash_bucket_size)
        return result

    def _count_lines(self, filepath: str) -> int:
        count = 0
        with open(filepath, "rb") as f:
            for _ in f:
                count += 1
        return count

    def _prepare_memmap(
        self,
        filepath: str,
        cache_dir: str = "data/cache",
        chunk_size: int = 500_000,
    ) -> Tuple[np.memmap, np.memmap, np.memmap]:
        """Read TSV in chunks and write to numpy memmap files."""
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)

        dense_path = cache / "dense.npy"
        sparse_path = cache / "sparse.npy"
        labels_path = cache / "labels.npy"

        if dense_path.exists() and sparse_path.exists() and labels_path.exists():
            logger.info("Loading cached memmap files...")
            n = np.load(cache / "meta.npy")[0]
            dense = np.memmap(dense_path, dtype=np.float32, mode="r", shape=(n, NUM_DENSE))
            sparse = np.memmap(sparse_path, dtype=np.int64, mode="r", shape=(n, NUM_SPARSE))
            labels = np.memmap(labels_path, dtype=np.float32, mode="r", shape=(n,))
            logger.info(f"Loaded {n:,} rows from cache")
            return dense, sparse, labels

        logger.info("Counting rows...")
        total = self._count_lines(filepath)
        if self.sample_size:
            total = min(total, self.sample_size)
        logger.info(f"Total rows: {total:,}")

        dense_mm = np.memmap(dense_path, dtype=np.float32, mode="w+", shape=(total, NUM_DENSE))
        sparse_mm = np.memmap(sparse_path, dtype=np.int64, mode="w+", shape=(total, NUM_SPARSE))
        labels_mm = np.memmap(labels_path, dtype=np.float32, mode="w+", shape=(total,))

        # First pass: compute dense mean/std with Welford's online algorithm
        logger.info("Pass 1/2: computing dense statistics...")
        count = 0
        mean = np.zeros(NUM_DENSE, dtype=np.float64)
        m2 = np.zeros(NUM_DENSE, dtype=np.float64)

        for chunk in pd.read_csv(
            filepath, sep="\t", header=None, names=ALL_COLS,
            chunksize=chunk_size, nrows=self.sample_size,
        ):
            vals = chunk[DENSE_COLS].fillna(0.0).values.astype(np.float64)
            for row in vals:
                count += 1
                delta = row - mean
                mean += delta / count
                delta2 = row - mean
                m2 += delta * delta2

        self.dense_mean = mean.astype(np.float32)
        self.dense_std = np.sqrt(m2 / count).astype(np.float32)
        self.dense_std[self.dense_std == 0] = 1.0
        logger.info("Dense stats computed")

        # Second pass: write normalized data to memmap
        logger.info("Pass 2/2: writing memmap files...")
        offset = 0
        for chunk in pd.read_csv(
            filepath, sep="\t", header=None, names=ALL_COLS,
            chunksize=chunk_size, nrows=self.sample_size,
        ):
            n_rows = len(chunk)
            end = offset + n_rows

            labels_mm[offset:end] = chunk[LABEL_COL].values.astype(np.float32)

            dense_vals = chunk[DENSE_COLS].fillna(0.0).values.astype(np.float32)
            dense_mm[offset:end] = (dense_vals - self.dense_mean) / self.dense_std

            sparse_raw = chunk[SPARSE_COLS].fillna(0).values.astype(np.int64)
            sparse_mm[offset:end] = self._encode_sparse_array(sparse_raw)

            offset = end
            if offset % (chunk_size * 10) == 0 or offset == total:
                logger.info(f"  {offset:,}/{total:,} rows written ({offset/total*100:.1f}%)")

        dense_mm.flush()
        sparse_mm.flush()
        labels_mm.flush()
        np.save(cache / "meta.npy", np.array([total]))
        logger.info(f"Memmap cache saved to {cache}")

        return dense_mm, sparse_mm, labels_mm

    def create_dataloaders(
        self,
        filepath: str,
        batch_size: int = 4096,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        num_workers: int = 4,
        seed: int = 42,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        dense, sparse, labels = self._prepare_memmap(filepath)
        total = len(labels)

        # Deterministic index-based split (avoids shuffling 45M indices in memory)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(total)

        test_n = int(total * test_ratio)
        val_n = int(total * val_ratio)
        train_n = total - test_n - val_n

        train_idx = indices[:train_n]
        val_idx = indices[train_n:train_n + val_n]
        test_idx = indices[train_n + val_n:]

        logger.info(f"Split — train: {train_n:,}, val: {val_n:,}, test: {test_n:,}")

        # Sort indices for sequential memmap access (better I/O performance)
        train_idx.sort()
        val_idx.sort()
        test_idx.sort()

        def make_loader(idx, shuffle):
            ds = CriteoMemmapDataset(dense[idx], sparse[idx], labels[idx])
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True)

        return make_loader(train_idx, True), make_loader(val_idx, False), make_loader(test_idx, False)

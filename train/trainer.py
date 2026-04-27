"""Training pipeline with AMP, early stopping, cosine annealing."""

import logging
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 3, mode: str = "max", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    def __init__(self, model: nn.Module, config: dict, device: str = None, model_name: str = None):
        self.model = model
        self.config = config
        self.model_name = model_name or model.__class__.__name__.lower()
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model.to(self.device)

        tc = config.get("train", {})
        self.epochs = tc.get("epochs", 10)
        self.use_amp = tc.get("use_amp", True) and self.device in ("cuda", "mps")
        self.amp_device = "cuda" if self.device == "cuda" else "cpu"

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=tc.get("lr", 1e-3), weight_decay=tc.get("weight_decay", 1e-5)
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=tc.get("T_0", 5), T_mult=2)
        self.scaler = GradScaler(self.amp_device, enabled=self.use_amp and self.device == "cuda")
        self.early_stopping = EarlyStopping(patience=tc.get("patience", 3), mode="max")
        self.max_grad_norm = tc.get("max_grad_norm", 5.0)

        self.checkpoint_dir = Path(tc.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_auc = 0.0
        self.history = []

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, n = 0.0, 0
        for dense, sparse, labels in loader:
            dense, sparse, labels = (
                dense.to(self.device),
                sparse.to(self.device),
                labels.to(self.device),
            )
            self.optimizer.zero_grad()
            with autocast(self.amp_device, enabled=self.use_amp):
                preds = self.model(dense, sparse)
                loss = self.criterion(preds, labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            n += 1
        self.scheduler.step()
        return {"train_loss": total_loss / n}

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss, n = 0.0, 0
        for dense, sparse, labels in loader:
            dense, sparse, labels = (
                dense.to(self.device),
                sparse.to(self.device),
                labels.to(self.device),
            )
            with autocast(self.amp_device, enabled=self.use_amp):
                preds = self.model(dense, sparse)
                loss = self.criterion(preds, labels)
            all_preds.append(torch.sigmoid(preds).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
            n += 1
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        return {
            "val_loss": total_loss / n,
            "auc": roc_auc_score(all_labels, all_preds),
            "logloss": log_loss(all_labels, np.clip(all_preds, 1e-7, 1 - 1e-7)),
        }

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        path = self.checkpoint_dir / f"best_{self.model_name}.pt"
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict(),
             "optimizer_state_dict": self.optimizer.state_dict(), "metrics": metrics},
            path,
        )
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str = None):
        path = path or self.checkpoint_dir / f"best_{self.model_name}.pt"
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path}")
        return ckpt["metrics"]

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        logger.info(f"Training on {self.device}, AMP={self.use_amp}, epochs={self.epochs}")
        best_metrics = {}
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_m = self._train_epoch(train_loader)
            val_m = self._evaluate(val_loader)
            elapsed = time.time() - t0
            record = {"epoch": epoch, **train_m, **val_m}
            self.history.append(record)
            logger.info(
                f"Epoch {epoch}/{self.epochs} ({elapsed:.1f}s) | "
                f"train_loss={train_m['train_loss']:.4f} | "
                f"val_loss={val_m['val_loss']:.4f} | AUC={val_m['auc']:.4f} | LogLoss={val_m['logloss']:.4f}"
            )
            if val_m["auc"] > self.best_auc:
                self.best_auc = val_m["auc"]
                best_metrics = record
                self._save_checkpoint(epoch, best_metrics)
            if self.early_stopping(val_m["auc"]):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        self.load_checkpoint()
        return best_metrics

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        metrics = self._evaluate(test_loader)
        logger.info(f"Test — AUC={metrics['auc']:.4f}, LogLoss={metrics['logloss']:.4f}")
        return metrics

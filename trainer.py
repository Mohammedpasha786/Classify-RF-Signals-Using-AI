Training engine for RF signal classifiers.
Supports:
  - Standard supervised training
  - Transfer learning (freeze backbone → fine-tune head)
  - SNR-stratified evaluation
  - TensorBoard / W&B logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


class Trainer:
    """
    Training engine for RF classifiers.

    Args:
        model:        PyTorch model
        train_loader: Training DataLoader
        val_loader:   Validation DataLoader
        n_epochs:     Total training epochs
        lr:           Peak learning rate
        weight_decay: AdamW weight decay
        device:       'cuda', 'mps', or 'cpu'
        log_dir:      Directory for TensorBoard logs and checkpoints
        use_wandb:    Enable Weights & Biases logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "auto",
        log_dir: str = "logs/",
        use_wandb: bool = False,
    ):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model  = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.n_epochs = n_epochs
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=n_epochs, eta_min=lr * 0.01)

        self.best_val_acc = 0.0
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }

        self._use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                print("wandb not installed — disabling W&B logging")
                self._use_wandb = False

        print(f"\nTrainer initialized")
        print(f"  Device  : {self.device}")
        print(f"  Epochs  : {n_epochs}")
        print(f"  LR      : {lr}")
        print(f"  Log dir : {self.log_dir}\n")

    # ------------------------------------------------------------------
    def train(self) -> Dict[str, List[float]]:
        """Full training loop."""
        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch()
            val_loss,   val_acc   = self._eval_epoch(self.val_loader)

            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{self.n_epochs}  "
                f"loss={train_loss:.4f}  acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                f"({elapsed:.1f}s)"
            )

            if self._use_wandb:
                self._wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss, "train/acc": train_acc,
                    "val/loss":   val_loss,   "val/acc":   val_acc,
                    "lr": self.scheduler.get_last_lr()[0],
                })

            # Save best checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint("best_model.pt")
                print(f"  ✓ New best val_acc={val_acc:.4f} — checkpoint saved")

        print(f"\nTraining complete. Best val_acc = {self.best_val_acc:.4f}")
        return self.history

    # ------------------------------------------------------------------
    def _train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        return total_loss / total, correct / total

    def _eval_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)

        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    def transfer_learn(
        self,
        new_n_classes: int,
        freeze_until: str = "layer2",
        fine_tune_lr: float = 1e-4,
        fine_tune_epochs: int = 20,
    ):
        """
        Perform transfer learning:
          1. Freeze backbone up to `freeze_until`
          2. Replace classifier head
          3. Fine-tune with low LR
        """
        if hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone(until_layer=freeze_until)
        if hasattr(self.model, "replace_classifier"):
            self.model.replace_classifier(new_n_classes)

        # Update optimizer with only trainable parameters
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=fine_tune_lr,
            weight_decay=1e-4,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=fine_tune_epochs, eta_min=fine_tune_lr * 0.01)
        self.n_epochs = fine_tune_epochs

        print(f"\n[Transfer Learning] Fine-tuning {new_n_classes} classes for {fine_tune_epochs} epochs")
        return self.train()

    # ------------------------------------------------------------------
    def save_checkpoint(self, filename: str):
        path = self.log_dir / filename
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc":         self.best_val_acc,
            "history":              self.history,
        }, path)

    def load_checkpoint(self, path: str) -> float:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Loaded checkpoint from {path}  (best_val_acc={self.best_val_acc:.4f})")
        return self.best_val_acc

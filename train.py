
Training entry point for RF signal classifier.

Usage:
    python scripts/train.py --config configs/resnet1d.yaml
    python scripts/train.py --model cnn1d --signals wifi_g bluetooth zigbee --snr-min -5
    python scripts/train.py --transfer --checkpoint checkpoints/base_model.pt --signals 5g_nr radar_lfm

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.iq_dataset import make_dataloaders
from src.data.signal_generator import SIGNAL_CLASSES
from src.models.classifier import build_model
from src.training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Train RF signal classifier")
    p.add_argument("--config",      type=str, default=None)
    p.add_argument("--model",       type=str, default="resnet1d",
                   choices=["cnn1d", "resnet1d", "cldnn"])
    p.add_argument("--signals",     nargs="+", default=SIGNAL_CLASSES)
    p.add_argument("--n-train",     type=int,   default=5000)
    p.add_argument("--n-val",       type=int,   default=1000)
    p.add_argument("--n-iq",        type=int,   default=1024)
    p.add_argument("--snr-min",     type=float, default=-10.0)
    p.add_argument("--snr-max",     type=float, default=30.0)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--log-dir",     type=str,   default="logs/")
    p.add_argument("--transfer",    action="store_true",
                   help="Fine-tune from a pretrained checkpoint")
    p.add_argument("--checkpoint",  type=str,   default=None)
    p.add_argument("--freeze-until",type=str,   default="layer2")
    p.add_argument("--wandb",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Load YAML config if provided
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        signals     = cfg.get("signals", SIGNAL_CLASSES)
        model_name  = cfg.get("model", args.model)
        n_train     = cfg.get("n_train", args.n_train)
        n_val       = cfg.get("n_val",   args.n_val)
        n_iq        = cfg.get("n_iq",    args.n_iq)
        snr_range   = (cfg.get("snr_min", args.snr_min), cfg.get("snr_max", args.snr_max))
        epochs      = cfg.get("epochs",   args.epochs)
        lr          = cfg.get("lr",       args.lr)
        batch_size  = cfg.get("batch_size", args.batch_size)
    else:
        signals     = args.signals
        model_name  = args.model
        n_train     = args.n_train
        n_val       = args.n_val
        n_iq        = args.n_iq
        snr_range   = (args.snr_min, args.snr_max)
        epochs      = args.epochs
        lr          = args.lr
        batch_size  = args.batch_size

    print(f"\n{'='*55}")
    print(f"  RF Signal Classifier — Training")
    print(f"{'='*55}")
    print(f"  Model    : {model_name}")
    print(f"  Signals  : {signals}")
    print(f"  SNR range: [{snr_range[0]}, {snr_range[1]}] dB")
    print(f"  Samples  : {n_iq} I/Q per example")
    print(f"  Transfer : {args.transfer}")
    print(f"{'='*55}\n")

    # Build dataloaders
    train_loader, val_loader, test_loader = make_dataloaders(
        signal_classes=signals,
        n_train=n_train,
        n_val=n_val,
        n_test=n_val // 2,
        n_iq=n_iq,
        snr_range=snr_range,
        batch_size=batch_size,
        num_workers=4,
        seed=args.seed,
    )

    # Build model
    model = build_model(model_name, n_classes=len(signals))

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=epochs,
        lr=lr,
        log_dir=args.log_dir,
        use_wandb=args.wandb,
    )

    # Load checkpoint for transfer learning
    if args.transfer and args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        trainer.transfer_learn(
            new_n_classes=len(signals),
            freeze_until=args.freeze_until,
            fine_tune_lr=lr * 0.1,
            fine_tune_epochs=epochs // 2,
        )
    else:
        trainer.train()

    print(f"\n✅ Done. Best val_acc = {trainer.best_val_acc:.4f}")
    print(f"   Checkpoint: {args.log_dir}/best_model.pt")


if __name__ == "__main__":
    main()

Evaluation metrics for RF signal classification:
  - Accuracy, precision, recall, F1 per class
  - Confusion matrix
  - Accuracy vs SNR curves
  - Top-K confidence analysis
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple


def classify_batch(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    top_k: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model on a batch, return predicted class indices and probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=-1)
        if top_k == 1:
            preds = probs.argmax(dim=1).cpu().numpy()
        else:
            preds = probs.topk(top_k, dim=1).indices.cpu().numpy()
    return preds, probs.cpu().numpy()


def compute_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Overall top-1 accuracy on a DataLoader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            preds, _ = classify_batch(model, x, device)
            correct += (preds == y.numpy()).sum()
            total += len(y)
    return {"accuracy": correct / total, "n_samples": total}


def confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute N×N confusion matrix (rows=true, cols=predicted)."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            preds, _ = classify_batch(model, x, device)
            for true, pred in zip(y.numpy(), preds):
                if 0 <= true < n_classes and 0 <= pred < n_classes:
                    cm[true, pred] += 1
    return cm


def per_class_metrics(cm: np.ndarray, class_names: Optional[List[str]] = None) -> Dict:
    """Precision, recall, F1 per class from confusion matrix."""
    n = cm.shape[0]
    results = []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        results.append({
            "class":     class_names[i] if class_names else str(i),
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
            "support":   int(cm[i, :].sum()),
        })
    overall_acc = cm.diagonal().sum() / cm.sum()
    return {"per_class": results, "overall_accuracy": float(overall_acc)}


def accuracy_vs_snr(
    model: nn.Module,
    signal_classes: List[str],
    snr_range: np.ndarray,
    n_per_snr: int = 500,
    n_iq: int = 1024,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, np.ndarray]:
    """
    Evaluate classification accuracy at each SNR point.
    Generates synthetic test data at each SNR and runs inference.

    Returns:
        dict with 'snr_db' and 'accuracy' arrays
    """
    from src.data.signal_generator import SignalDatasetGenerator, GeneratorConfig, CLASS_TO_IDX

    model.eval()
    cfg = GeneratorConfig(n_samples=n_iq, seed=999)
    gen = SignalDatasetGenerator(cfg)
    accs = []

    for snr_db in snr_range:
        X, y = gen.generate_batch(signal_classes, n_per_snr // len(signal_classes), (snr_db, snr_db))
        X_t = torch.from_numpy(X)
        preds, _ = classify_batch(model, X_t, device)
        acc = (preds == y).mean()
        accs.append(float(acc))
        print(f"  SNR={snr_db:5.1f} dB  acc={acc:.3f}")

    return {"snr_db": snr_range, "accuracy": np.array(accs)}


def print_classification_report(metrics: Dict, title: str = "Classification Report"):
    """Pretty-print per-class metrics table."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"{'─'*60}")
    print(f"  {'Class':<18} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print(f"{'─'*60}")
    for row in metrics["per_class"]:
        print(
            f"  {row['class']:<18} "
            f"{row['precision']:>10.3f} "
            f"{row['recall']:>8.3f} "
            f"{row['f1']:>8.3f} "
            f"{row['support']:>9d}"
        )
    print(f"{'='*60}\n")

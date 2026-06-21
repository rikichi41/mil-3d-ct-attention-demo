from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import nn

from .data import SyntheticBagConfig, make_dataloaders
from .models import AttentionMILClassifier, MaxMILClassifier


def build_model(model_name: str) -> nn.Module:
    if model_name == "max":
        return MaxMILClassifier()
    if model_name == "attention":
        return AttentionMILClassifier(gated=False)
    if model_name == "gated_attention":
        return AttentionMILClassifier(gated=True)
    raise ValueError(f"Unknown model: {model_name}")


def iter_bag_logits(model: nn.Module, bags: Iterable[torch.Tensor], device: torch.device):
    logits = []
    weights = []
    for bag in bags:
        logit, attention_weights = model(bag.to(device))
        logits.append(logit)
        weights.append(attention_weights.detach().cpu())
    return torch.stack(logits), weights


def evaluate(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for bags, labels, _ in loader:
            logits, _ = iter_bag_logits(model, bags, device)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().astype(int).tolist())

    y_true = np.asarray(all_labels)
    y_prob = np.asarray(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def train_one_epoch(model: nn.Module, loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    losses = []
    for bags, labels, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        logits, _ = iter_bag_logits(model, bags, device)
        loss = criterion(logits, labels.to(device))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def load_config(path: str | None, args) -> SyntheticBagConfig:
    values = {}
    if path:
        values.update(json.loads(Path(path).read_text()))
    for key in ["n_patients", "positive_fraction", "min_blocks", "max_blocks", "depth", "height", "width", "seed"]:
        value = getattr(args, key, None)
        if value is not None:
            values[key] = value
    return SyntheticBagConfig(**values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a privacy-safe synthetic 3D MIL demo.")
    parser.add_argument("--config", default="configs/synthetic_demo.json")
    parser.add_argument("--model", choices=["max", "attention", "gated_attention"], default="attention")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu", help="Use 'cpu', 'cuda', or 'auto'.")
    parser.add_argument("--n_patients", type=int, default=None)
    parser.add_argument("--positive_fraction", type=float, default=None)
    parser.add_argument("--min_blocks", type=int, default=None)
    parser.add_argument("--max_blocks", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed if args.seed is not None else 7)
    config = load_config(args.config, args)
    device = resolve_device(args.device)

    train_loader, test_loader = make_dataloaders(config, batch_size=args.batch_size)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print("Synthetic 3D MIL demo")
    print("model:", args.model)
    print("device:", device)
    print("data:", asdict(config))

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, test_loader, device)
        print(
            f"epoch {epoch:03d} "
            f"loss={loss:.4f} "
            f"roc_auc={metrics['roc_auc']:.4f} "
            f"pr_auc={metrics['pr_auc']:.4f} "
            f"accuracy={metrics['accuracy']:.4f}"
        )

    print("final_metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

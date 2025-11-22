"""Evaluate a trained checkpoint on a given split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from src.data import DataSettings, build_transforms
from src.model import build_model
from src.utils import load_checkpoint, load_class_mapping, topk_accuracy

matplotlib.use("Agg")
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cat breed classifier checkpoints.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Root processed dataset directory.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint (.pt).")
    parser.add_argument("--class-map", type=Path, default=None, help="Optional path to class_to_idx.json.")
    parser.add_argument("--model-name", type=str, default=None, help="Backbone architecture (defaults to checkpoint metadata).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--report-path", type=Path, default=None, help="Where to store aggregated metrics JSON.")
    parser.add_argument("--cm-path", type=Path, default=None, help="Path for confusion matrix PNG.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    metadata = checkpoint.get("metadata", {})

    class_map = (
        load_class_mapping(args.class_map)
        if args.class_map
        else checkpoint.get("class_to_idx")
    )
    if not class_map:
        raise ValueError("class_to_idx missing. Pass --class-map or train with newer scripts.")
    idx_to_class = {idx: cls for cls, idx in class_map.items()}

    model_name = args.model_name or metadata.get("model_name") or "efficientnet_b0"
    dropout = metadata.get("dropout", 0.3)
    model = build_model(
        model_name=model_name,
        num_classes=len(class_map),
        pretrained=False,
        dropout=dropout,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    settings = DataSettings(image_size=args.image_size, center_crop=args.crop_size, augment=False)
    _, eval_tfms = build_transforms(settings)
    split_dir = args.data_dir / args.split
    dataset = datasets.ImageFolder(split_dir, transform=eval_tfms)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    metrics = evaluate_model(model, dataloader, device, args.topk, idx_to_class)

    report_path = Path(args.report_path or args.checkpoint.parent / f"evaluation-{args.split}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2))
    console.print(f"[bold green]Saved metrics to {report_path}")

    cm_path = Path(args.cm_path or args.checkpoint.parent / f"confusion-{args.split}.png")
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(metrics["confusion_matrix"], idx_to_class, cm_path)
    console.print(f"[bold green]Saved confusion matrix to {cm_path}")


def evaluate_model(model, dataloader, device, topk: List[int], idx_to_class: dict) -> dict:
    all_preds = []
    all_labels = []
    total = 0
    correct = 0
    topk_totals = {k: 0.0 for k in topk}

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total += batch_size
            correct += (preds == labels).sum().item()

            batch_topk = topk_accuracy(logits, labels, topk=tuple(topk))
            for k, acc in zip(topk, batch_topk):
                topk_totals[k] += acc * batch_size

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    cls_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, digits=4)
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "accuracy": correct / max(total, 1),
        "top_k": {f"top_{k}": topk_totals[k] / max(total, 1) for k in topk},
        "classification_report": cls_report,
        "confusion_matrix": cm,
    }
    return metrics


def save_confusion_matrix(cm: List[List[int]], idx_to_class: dict, path: Path) -> None:
    matrix = np.array(cm)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(idx_to_class))
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    ax.set(
        xticks=tick_marks,
        yticks=tick_marks,
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

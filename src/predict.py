"""Inference helper for cat breed classification checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import torch
from PIL import Image
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, Dataset

from src.data import DataSettings, build_transforms
from src.model import build_model
from src.utils import load_checkpoint, load_class_mapping

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on cat images to predict breeds.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained checkpoint.")
    parser.add_argument("--class-map", type=Path, default=None, help="Optional class mapping JSON.")
    parser.add_argument("--model-name", type=str, default=None, help="Backbone architecture (defaults to checkpoint metadata).")
    parser.add_argument("--inputs", nargs="+", required=True, help="Image files or folders containing images.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path to save predictions as JSON.")
    parser.add_argument("--save-csv", type=Path, default=None, help="Optional path to save predictions as CSV.")
    return parser.parse_args()


class ImageDataset(Dataset):
    def __init__(self, paths: List[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load image {path}") from exc
        return self.transform(image), str(path)


def gather_images(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            for file in path.rglob("*"):
                if file.suffix.lower() in IMAGE_EXTS:
                    paths.append(file)
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            paths.append(path)
    if not paths:
        raise ValueError("No valid image files found.")
    return sorted(paths)


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
        raise ValueError("class_to_idx missing. Pass --class-map or use checkpoints produced by this repo.")
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

    _, eval_tfms = build_transforms(DataSettings(image_size=args.image_size, center_crop=args.crop_size, augment=False))
    image_paths = gather_images(args.inputs)
    dataset = ImageDataset(image_paths, eval_tfms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    results = run_inference(model, dataloader, device, idx_to_class, args.topk)
    render_table(results)
    maybe_save(results, args.save_json, args.save_csv, args.topk)


def run_inference(model, dataloader, device, idx_to_class, topk: int):
    effective_topk = min(topk, len(idx_to_class))
    all_outputs = []
    with torch.no_grad():
        for batch_images, paths in dataloader:
            batch_images = batch_images.to(device, non_blocking=True)
            logits = model(batch_images)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_indices = probs.topk(effective_topk, dim=1)
            for path, row_probs, row_indices in zip(paths, top_probs, top_indices):
                predictions = [
                    {"label": idx_to_class[idx.item()], "confidence": float(prob.item())}
                    for idx, prob in zip(row_indices, row_probs)
                ]
                all_outputs.append({"path": path, "predictions": predictions})
    return all_outputs


def render_table(results) -> None:
    table = Table(title="Cat Breed Predictions")
    table.add_column("Image", justify="left")
    table.add_column("Top prediction", justify="left")
    table.add_column("Top-k details", justify="left")
    for item in results:
        best = item["predictions"][0]
        topk_str = ", ".join(f"{pred['label']} ({pred['confidence']:.1%})" for pred in item["predictions"])
        table.add_row(Path(item["path"]).name, f"{best['label']} ({best['confidence']:.1%})", topk_str)
    console.print(table)
    console.print(f"[bold]{len(results)}[/bold] predictions generated.")


def maybe_save(results, json_path: Path | None, csv_path: Path | None, topk: int) -> None:
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2))
        console.print(f"[green]Saved JSON predictions to {json_path}")
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        header = ["path"]
        max_preds = min(topk, len(results[0]["predictions"])) if results else topk
        for rank in range(1, max_preds + 1):
            header.extend([f"top{rank}_label", f"top{rank}_confidence"])
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for item in results:
                row = [item["path"]]
                for pred in item["predictions"]:
                    row.extend([pred["label"], f"{pred['confidence']:.6f}"])
                writer.writerow(row)
        console.print(f"[green]Saved CSV predictions to {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Utility to build stratified train/val/test splits from a folder-per-class dataset."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a folder-per-class image dataset into train/val/test subsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, required=True, help="Root folder containing <class> subfolders.")
    parser.add_argument("--destination", type=Path, required=True, help="Output folder for train/val/test splits.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of images for validation.")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Fraction of images for testing.")
    parser.add_argument("--min-images", type=int, default=20, help="Skip classes with fewer images than this threshold.")
    parser.add_argument("--max-per-class", type=int, default=0, help="Down-sample classes above this many images (0 = keep all).")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would happen without copying files.")
    return parser.parse_args()


def discover_images(source: Path) -> Dict[str, List[Path]]:
    class_to_images: Dict[str, List[Path]] = {}
    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        images = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if images:
            class_to_images[class_dir.name] = images
    return class_to_images


def split_indices(total: int, val_ratio: float, test_ratio: float) -> Dict[str, slice]:
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)
    train_count = total - val_count - test_count
    return {
        "train": slice(0, train_count),
        "val": slice(train_count, train_count + val_count),
        "test": slice(train_count + val_count, total),
    }


def copy_subset(files: List[Path], dest_dir: Path, dry_run: bool) -> None:
    if not files:
        return
    if dry_run:
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        target = dest_dir / src.name
        if target.exists():
            continue
        shutil.copy2(src, target)


def perform_split(
    source: Path,
    destination: Path,
    val_ratio: float,
    test_ratio: float,
    min_images: int,
    max_per_class: int,
    seed: int,
    dry_run: bool,
) -> Dict[str, Dict[str, int]]:
    assert 0 <= val_ratio < 1, "Validation ratio must be in [0, 1)."
    assert 0 <= test_ratio < 1, "Test ratio must be in [0, 1)."
    assert val_ratio + test_ratio < 0.9, "Keep at least 10% for training."

    rng = random.Random(seed)
    class_to_images = discover_images(source)

    if not class_to_images:
        raise SystemExit(f"No class folders with images found under {source}")

    summary = []
    for class_name, paths in tqdm(class_to_images.items(), desc="Splitting classes"):
        rng.shuffle(paths)
        if len(paths) < min_images:
            continue
        if max_per_class > 0:
            paths = paths[:max_per_class]
        slices = split_indices(len(paths), val_ratio, test_ratio)
        per_split = {split: paths[idx] for split, idx in slices.items()}
        summary.append((class_name, {k: len(v) for k, v in per_split.items()}))
        if not dry_run:
            for split, files in per_split.items():
                dest_dir = destination / split / class_name
                copy_subset(list(files), dest_dir, dry_run)

    summary_str = "\n".join(
        f"{cls}: train={counts['train']}, val={counts['val']}, test={counts['test']}" for cls, counts in summary
    )
    print(f"Completed splitting {len(summary)} classes.\n{summary_str}")
    return {cls: counts for cls, counts in summary}


def main() -> None:
    args = parse_args()
    perform_split(
        source=args.source,
        destination=args.destination,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_images=args.min_images,
        max_per_class=args.max_per_class,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

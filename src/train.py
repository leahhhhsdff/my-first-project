"""Command-line training script for cat breed classification."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
from rich.console import Console
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data import DataSettings, compute_class_weights, create_dataloaders, create_datasets
from src.model import build_model
from src.utils import AverageMeter, EarlyStopping, ensure_dir, save_checkpoint, save_class_mapping, save_history, set_seed

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a cat breed classifier with PyTorch.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Directory with train/val[/test] splits.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to store checkpoints & logs.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-name", type=str, default="efficientnet_b0")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained ImageNet weights.")
    parser.add_argument("--no-augment", action="store_true", help="Disable heavy data augmentation.")
    parser.add_argument("--auto-augment", action="store_true", help="Enable torchvision AutoAugment.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--class-weighting", action="store_true", help="Use inverse-frequency class weights.")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum delta for early stopping.")
    parser.add_argument("--device", type=str, default=None, help="Override autodetected device (cpu|cuda).")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    console.print(f"[bold]Using device:[/bold] {device}")

    timestamp = datetime.now()
    run_dir = args.output_dir or Path("artifacts") / f"run-{timestamp:%Y%m%d-%H%M%S}"
    run_dir = ensure_dir(Path(run_dir))
    console.print(f"[bold]Artifacts:[/bold] {run_dir}")
    config_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config_dict["resolved_device"] = str(device)
    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    data_settings = DataSettings(
        image_size=args.image_size,
        center_crop=args.crop_size,
        augment=not args.no_augment,
        auto_augment=args.auto_augment,
    )
    datasets = create_datasets(args.data_dir, data_settings)
    dataloaders = create_dataloaders(datasets, args.batch_size, args.num_workers, pin_memory=device.type == "cuda")
    class_to_idx = datasets["train"].class_to_idx

    model = build_model(
        args.model_name,
        num_classes=len(class_to_idx),
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    class_weights = None
    if args.class_weighting:
        class_weights = compute_class_weights(datasets["train"]).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=args.mixed_precision and device.type == "cuda")
    early_stopper = EarlyStopping(args.patience, args.min_delta) if args.early_stopping else None

    best_val_acc = 0.0
    history: list[Dict] = []

    for epoch in range(1, args.epochs + 1):
        console.print(f"\n[bold yellow]Epoch {epoch}/{args.epochs}[/bold yellow]")
        train_loss, train_acc = run_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            device,
            scaler,
            args.grad_clip,
            training=True,
            use_amp=scaler.is_enabled(),
        )
        val_loss, val_acc = run_epoch(
            model,
            dataloaders.get("val"),
            criterion,
            optimizer,
            device,
            scaler,
            args.grad_clip,
            training=False,
            use_amp=scaler.is_enabled(),
        )

        scheduler.step()
        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_stats)
        console.print(json.dumps(epoch_stats, indent=2))

        metadata = {
            "model_name": args.model_name,
            "dropout": args.dropout,
            "num_classes": len(class_to_idx),
            "timestamp": timestamp.isoformat(),
        }
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                run_dir / "best.pt",
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                best_val_acc,
                class_to_idx,
                metadata=metadata,
            )
        save_checkpoint(
            run_dir / "last.pt",
            model.state_dict(),
            optimizer.state_dict(),
            epoch,
            best_val_acc,
            class_to_idx,
            metadata=metadata,
        )
        save_history(history, run_dir / "history.json")
        save_class_mapping(class_to_idx, run_dir / "class_to_idx.json")

        if early_stopper and early_stopper.step(val_acc):
            console.print("[bold red]Early stopping triggered.[/bold red]")
            break

    console.print(f"[bold green]Training complete.[/bold green] Best val accuracy: {best_val_acc:.4f}")


def run_epoch(
    model: torch.nn.Module,
    dataloader,
    criterion,
    optimizer,
    device,
    scaler,
    grad_clip,
    training: bool,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train(mode=training)
    loss_meter = AverageMeter("loss")
    correct = 0
    total = 0
    if dataloader is None:
        return 0.0, 0.0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if training:
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)

        loss_meter.update(loss.item(), images.size(0))
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return loss_meter.avg, correct / max(total, 1)


if __name__ == "__main__":
    main()

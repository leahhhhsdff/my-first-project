"""Generic helpers for training and evaluation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_history(history: List[Dict], output_path: Path) -> None:
    output_path.write_text(json.dumps(history, indent=2))


def save_class_mapping(class_to_idx: Dict[str, int], output_path: Path) -> None:
    output_path.write_text(json.dumps(class_to_idx, indent=2))


def load_class_mapping(path: Path) -> Dict[str, int]:
    return json.loads(path.read_text())


@dataclass
class AverageMeter:
    name: str
    value: float = 0.0
    total: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.value = val
        self.total += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float("-inf")

    def step(self, metric: float) -> bool:
        if metric > self.best_score + self.min_delta:
            self.best_score = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk: Sequence[int] = (1,)) -> List[float]:
    with torch.no_grad():
        max_k = max(topk)
        _, pred = output.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k * (1.0 / target.size(0))).item())
        return res


def save_checkpoint(
    path: Path,
    model_state: Dict,
    optimizer_state: Dict,
    epoch: int,
    best_metric: float,
    class_to_idx: Dict[str, int],
    metadata: Dict | None = None,
) -> None:
    payload = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "epoch": epoch,
        "best_metric": best_metric,
        "class_to_idx": class_to_idx,
    }
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict:
    return torch.load(path, map_location=map_location)

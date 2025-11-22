# Detecting & Classifying Cat Breeds

An end‑to‑end transfer learning pipeline for building a cat breed recognizer. The repository is framework-agnostic about the dataset source, but it is optimized for the `images.cv` export (≈18k photos across dozens of breeds) mentioned in the brief. The code covers dataset preparation, model fine-tuning, evaluation, and inference on arbitrary images.

---

## Why this project?
- **Pet lovers & professionals** can quickly build a light-weight breed classifier without starting from scratch.
- **Reproducible pipeline**: deterministic splits, experiment tracking via JSON, and CLI tooling.
- **Modern CV stack** built with PyTorch + Torchvision, ready for GPUs yet runnable on CPU for debugging.

---

## Project layout

```
├── data/                 # Place datasets here (ignored by git)
│   ├── raw/              # Unmodified dump from images.cv (zip contents)
│   └── processed/        # Output of scripts/split_dataset.py (train/val/test)
├── artifacts/            # Checkpoints, logs, metrics
├── scripts/
│   └── split_dataset.py  # Deterministic stratified split helper
├── src/
│   ├── data.py           # Datasets + transforms + dataloader builders
│   ├── model.py          # Transfer-learning ready model factory
│   ├── train.py          # Main training loop
│   ├── eval.py           # Evaluate checkpoints on held-out data
│   └── predict.py        # Batch/single-image inference utility
├── tests/                # Lightweight unit tests (pytest)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Torch/Torchvision wheels are GPU-aware; if you need CUDA specific builds, follow the [official instructions](https://pytorch.org/get-started/locally/) before installing the remaining packages.

---

## Preparing the data

1. **Download from images.cv**  
   - Follow <https://images.cv/how-to-use> and request the *cat breed* dataset.  
   - Place the unzipped folder under `data/raw/`. The folder should already contain sub-folders per breed (`data/raw/<breed_name>/*.jpg`).

2. **Create deterministic train/val/test splits**

   ```bash
   python scripts/split_dataset.py \
       --source data/raw \
       --destination data/processed \
       --val-ratio 0.15 \
       --test-ratio 0.10 \
       --min-images 20
   ```

   The script copies files into `processed/train|val|test/<breed>/...` while skipping breeds that do not satisfy `--min-images`.

3. (Optional) **Balance long-tail classes**  
   You can provide `--max-per-class` to down-sample over-represented breeds for fairer training.

> **Note:** The repo intentionally excludes data files; ensure you have at least a few images per breed before attempting to train.

---

## Training

```bash
python -m src.train \
    --data-dir data/processed \
    --output-dir artifacts/exp1 \
    --model-name efficientnet_b0 \
    --epochs 20 \
    --batch-size 48 \
    --lr 3e-4 \
    --use-pretrained
```

Key options:

| Flag | Description |
| --- | --- |
| `--model-name` | Backbone (`efficientnet_b0`, `resnet50`, `convnext_tiny`) |
| `--freeze-backbone` | Freeze feature extractor for fast linear probing |
| `--mixed-precision` | Enables `torch.cuda.amp` for speed/memory gains |
| `--early-stopping` | Stop after N stagnant epochs (defaults to 5) |
| `--class-weighting` | Use inverse-frequency weights to handle imbalance |

Outputs saved under `artifacts/exp*/`:
- `best.pt` – best scoring checkpoint
- `last.pt` – final epoch weights
- `history.json` – metrics per epoch
- `class_to_idx.json` – breed ↔ id mapping

---

## Evaluation

```bash
python -m src.eval \
    --data-dir data/processed/test \
    --checkpoint artifacts/exp1/best.pt \
    --class-map artifacts/exp1/class_to_idx.json
```

The script reports accuracy, top‑k metrics, per-class precision/recall/F1, and writes a confusion matrix plot to the output directory.

---

## Inference

```bash
python -m src.predict \
    --checkpoint artifacts/exp1/best.pt \
    --class-map artifacts/exp1/class_to_idx.json \
    --inputs path/to/photo1.jpg path/to/folder_of_images \
    --topk 3
```

Predictions stream to stdout via a Rich table and can optionally be exported as JSON/CSV (`--save-json`, `--save-csv`).

---

## Testing

```bash
pytest -q
```

Unit tests rely on synthetic images and finish quickly; they exist to prevent regressions in transforms and dataset utilities.

---

## Next steps & ideas
- Add automatic mixed-precision gradient scaling for CPUs via `torch.autocast("cpu")`.
- Integrate Weights & Biases or MLflow for experiment tracking.
- Distill the model for on-device inference (e.g., via TorchScript or ONNX).
- Add active learning hooks to prioritize misclassified breeds.

Enjoy building with images.cv! Contributions and issues are welcome.

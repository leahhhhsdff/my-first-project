import numpy as np
import torch
from PIL import Image

from src.data import DataSettings, build_transforms, compute_class_weights


def test_eval_transform_output_shape():
    settings = DataSettings(image_size=128, center_crop=112, augment=False)
    _, eval_tfms = build_transforms(settings)
    image = Image.fromarray(np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8))
    tensor = eval_tfms(image)
    assert tensor.shape == (3, 112, 112)


def test_compute_class_weights_balances_counts():
    class DummyDataset:
        classes = ["a", "b"]
        samples = [("x", 0)] * 2 + [("y", 1)] * 6

    weights = compute_class_weights(DummyDataset())
    assert torch.isclose(weights[0] / weights[1], torch.tensor(3.0), atol=1e-3)

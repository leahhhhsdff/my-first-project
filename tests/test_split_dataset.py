from pathlib import Path

from PIL import Image

from scripts.split_dataset import perform_split, split_indices


def test_split_indices_counts_add_up():
    slices = split_indices(100, 0.1, 0.2)
    assert slices["train"].stop - slices["train"].start == 70
    assert slices["val"].stop - slices["val"].start == 10
    assert slices["test"].stop - slices["test"].start == 20


def test_perform_split_creates_expected_structure(tmp_path):
    source = tmp_path / "raw"
    dest = tmp_path / "processed"
    for cls, count in {"abyssinian": 30, "bengal": 40}.items():
        class_dir = source / cls
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            image = Image.new("RGB", (16, 16), color=(idx % 255, 0, 0))
            image.save(class_dir / f"{idx}.jpg")

    summary = perform_split(
        source=source,
        destination=dest,
        val_ratio=0.1,
        test_ratio=0.2,
        min_images=5,
        max_per_class=0,
        seed=123,
        dry_run=False,
    )

    assert set(summary.keys()) == {"abyssinian", "bengal"}
    for cls, counts in summary.items():
        for split, expected in counts.items():
            split_dir = dest / split / cls
            assert split_dir.exists()
            actual = len(list(split_dir.glob("*.jpg")))
            assert actual == expected

"""
Split labeled images from golden_data into train/val/test folder structure.

Only images that have a corresponding non-empty .txt label file are included.
Unlabeled images are ignored.

Usage:
    python split_dataset.py
"""

import random
import shutil
from pathlib import Path

GOLDEN_DATA = Path("./dataset/golden_data")
OUTPUT     = Path("./dataset")
SPLITS     = {"train": 0.80, "val": 0.15, "test": 0.05}
SEED       = 42


def main():
    labeled = [
        p for p in sorted(GOLDEN_DATA.glob("*.jpg"))
        if p.with_suffix(".txt").exists()
        and p.with_suffix(".txt").stat().st_size > 0
    ]

    if not labeled:
        raise SystemExit("No labeled images found in " + str(GOLDEN_DATA))

    random.seed(SEED)
    random.shuffle(labeled)

    n = len(labeled)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])

    split_images = {
        "train": labeled[:n_train],
        "val":   labeled[n_train:n_train + n_val],
        "test":  labeled[n_train + n_val:],
    }

    for split, images in split_images.items():
        img_dir = OUTPUT / "images" / split
        lbl_dir = OUTPUT / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in images:
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(img.with_suffix(".txt"), lbl_dir / img.with_suffix(".txt").name)

        print(f"  {split:<6} {len(images):>4} images")

    print(f"\nTotal: {n} labeled images split into train/val/test")
    print(f"Dataset ready at: {OUTPUT.resolve()}")
    print("\nNext: python train.py")


if __name__ == "__main__":
    main()

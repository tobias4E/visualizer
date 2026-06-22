"""
Step 1b – Auto-label EV charger images locally using Grounding DINO.

Usage:
    python autolabel.py --images ./my_images --output ./dataset

This writes YOLO-format .txt label files and splits the dataset into
train / val / test automatically. No data leaves your machine.

Requires:
    pip install autodistill autodistill-grounding-dino
"""

import argparse
import random
import shutil
from pathlib import Path


def split_dataset(images: list[Path], output_dir: Path, train=0.80, val=0.15):
    """Copy images + labels into train/val/test folder structure."""
    random.shuffle(images)
    n = len(images)
    n_train = int(n * train)
    n_val = int(n * val)

    splits = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    for split, split_images in splits.items():
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            lbl_path = img_path.with_suffix(".txt")
            shutil.copy2(img_path, img_dir / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dir / lbl_path.name)
            else:
                # Image got no detections — write empty label file
                (lbl_dir / lbl_path.name).touch()

    return {split: len(imgs) for split, imgs in splits.items()}


def write_data_yaml(output_dir: Path):
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "test:  images/test\n"
        "\n"
        "nc: 1\n"
        "names: ['charger']\n"
    )
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Auto-label EV charger images locally")
    parser.add_argument(
        "--images",
        required=True,
        help="Folder containing your raw images (jpg/png/webp)",
    )
    parser.add_argument(
        "--output",
        default="./dataset",
        help="Output folder for the labeled dataset (default: ./dataset)",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for Grounding DINO detections (default: 0.25)",
    )
    parser.add_argument(
        "--ext",
        default=".jpg",
        help="Image extension to look for (default: .jpg)",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Skip train/val/test split — just write labels next to images",
    )
    args = parser.parse_args()

    images_dir = Path(args.images).resolve()
    output_dir = Path(args.output).resolve()

    if not images_dir.exists():
        raise SystemExit(f"Images folder not found: {images_dir}")

    # Import here so missing deps give a clear error message
    try:
        from autodistill_grounding_dino import GroundingDINO
        from autodistill.detection import CaptionOntology
    except ImportError:
        raise SystemExit(
            "Missing dependencies. Run:\n"
            "  pip install autodistill autodistill-grounding-dino"
        )

    print(f"Images : {images_dir}")
    print(f"Output : {output_dir}")
    print(f"Threshold: {args.box_threshold}\n")

    # Grounding DINO runs fully locally — no upload, no API key
    model = GroundingDINO(
        ontology=CaptionOntology({"EV charging station": "charger"}),
        box_threshold=args.box_threshold,
    )

    # Label all images; writes <image>.txt next to each image
    print("Running Grounding DINO auto-labeling...")
    model.label(str(images_dir), extension=args.ext)

    if args.no_split:
        print("\nDone. Labels written next to images (no split).")
        return

    # Collect only images that exist (support jpg/jpeg/png/webp regardless of --ext)
    exts = {args.ext.lower(), ".jpg", ".jpeg", ".png", ".webp"}
    all_images = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]

    if not all_images:
        raise SystemExit(f"No images found in {images_dir}")

    print(f"\nFound {len(all_images)} images. Splitting into train/val/test...")
    counts = split_dataset(all_images, output_dir)
    yaml_path = write_data_yaml(output_dir)

    print(f"\n{'Split':<8} {'Images':>6}")
    print("-" * 16)
    for split, count in counts.items():
        print(f"{split:<8} {count:>6}")

    print(f"\nDataset written to: {output_dir}")
    print(f"Config file:        {yaml_path}")
    print("\nReview labels with:  pip install labelImg && labelImg")
    print("Then train with:     python train.py")


if __name__ == "__main__":
    main()

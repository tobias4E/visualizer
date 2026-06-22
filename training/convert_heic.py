"""
Convert all HEIC images in a folder to JPG in-place.

Usage:
    python convert_heic.py [folder]

Default folder: ./dataset/golden_data

Requires:
    pip install pillow pillow-heif
"""

import argparse
import sys
from pathlib import Path

try:
    import pillow_heif
    from PIL import Image
    pillow_heif.register_heif_opener()
except ImportError:
    sys.exit("Missing dependencies. Run:\n  pip install pillow pillow-heif")


def convert(folder: Path):
    heic_files = sorted(folder.glob("*.HEIC")) + sorted(folder.glob("*.heic"))

    if not heic_files:
        print(f"No HEIC files found in {folder}")
        return

    print(f"Converting {len(heic_files)} HEIC files in {folder} ...")

    ok, failed = 0, []
    for src in heic_files:
        dst = src.with_suffix(".jpg")
        try:
            with Image.open(src) as img:
                img.convert("RGB").save(dst, "JPEG", quality=95)
            src.unlink()
            ok += 1
            print(f"  {src.name} → {dst.name}")
        except Exception as e:
            failed.append((src.name, str(e)))
            print(f"  FAILED {src.name}: {e}")

    print(f"\nDone: {ok} converted, {len(failed)} failed.")
    if failed:
        for name, err in failed:
            print(f"  {name}: {err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folder",
        nargs="?",
        default="./dataset/golden_data",
        help="Folder containing HEIC files (default: ./dataset/golden_data)",
    )
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.exists():
        sys.exit(f"Folder not found: {folder}")

    convert(folder)


if __name__ == "__main__":
    main()

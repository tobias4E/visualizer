"""
Step 4 – Export the best checkpoint to TF.js for use in the browser app.

Produces best_web_model/ next to the weights file.
Uses imgsz=320 (half of training resolution) for fast phone-browser inference.
Requires: pip install ultralytics>=8.3
"""

import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument(
    "--weights",
    default="runs/detect/ev_charger_v1/weights/best.pt",
    help="Path to trained .pt weights file",
)
parser.add_argument(
    "--imgsz",
    type=int,
    default=320,
    help="Inference image size for the exported model (320 recommended for browsers)",
)
args = parser.parse_args()

model = YOLO(args.weights)
model.export(format="tfjs", imgsz=args.imgsz)

import pathlib
weights_dir = pathlib.Path(args.weights).parent
web_model_dir = weights_dir / "best_web_model"

print("\nExport complete.")
print(f"TF.js model files: {web_model_dir}")
print("Copy that folder next to index.html and update the tf.loadGraphModel() path.")

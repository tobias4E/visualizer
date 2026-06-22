"""
Step 3 – Evaluate the trained model on the held-out test split.

Prints mAP50, mAP50-95, precision, and recall.
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
    "--data",
    default="dataset/data.yaml",
    help="Path to data.yaml",
)
parser.add_argument(
    "--split",
    default="test",
    choices=["val", "test"],
    help="Dataset split to evaluate on",
)
args = parser.parse_args()

model = YOLO(args.weights)
metrics = model.val(data=args.data, split=args.split)

print("\n--- Evaluation results ---")
print(f"mAP50:    {metrics.box.map50:.3f}  (target > 0.65, good > 0.80)")
print(f"mAP50-95: {metrics.box.map:.3f}   (target > 0.40, good > 0.60)")

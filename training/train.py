"""
Step 2 – Fine-tune YOLOv8n on the EV charger dataset.

Run after labeling is complete and dataset/ has images/ + labels/ populated.
Requires: pip install ultralytics>=8.3
"""

from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data=str(Path(__file__).parent / "dataset" / "data.yaml"),
    epochs=100,
    imgsz=640,
    batch=4,       # low batch size for CPU training
    device="cpu",
    name="ev_charger_v1",
    patience=20,
    # Extra brightness/contrast jitter for dawn/dusk/indoor conditions
    hsv_h=0.02,
    hsv_s=0.8,
    hsv_v=0.5,
    degrees=5,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
)

print("\nTraining complete.")
print("Best weights: runs/detect/ev_charger_v1/weights/best.pt")

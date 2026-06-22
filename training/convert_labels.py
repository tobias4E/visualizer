"""
Convert X-AnyLabeling JSON files to YOLO bounding box .txt files.

Works with both rectangle and polygon shapes — polygons are converted
to their bounding box automatically.

Usage:
    python convert_labels.py [folder]

Default folder: ./dataset/golden_data
"""

import json
import sys
from pathlib import Path


# Map any label name variant to the single class id
LABEL_MAP = {
    "charger": 0,
    "charging_station": 0,
    "charging_station_no_cable": 0,
    "wallbox": 0,
    "ev charger": 0,
    "ev charging station": 0,
    "electric vehicle charging station": 0,
}


def polygon_to_bbox(points):
    """Return (x_min, y_min, x_max, y_max) from a list of [x, y] points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def convert(folder: Path):
    json_files = sorted(folder.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {folder}")
        return

    ok, skipped, no_shapes = 0, 0, 0

    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)

        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        shapes = data.get("shapes", [])

        if not shapes:
            no_shapes += 1
            # Write empty label file so YOLO knows this image has no objects
            json_path.with_suffix(".txt").write_text("")
            continue

        lines = []
        for shape in shapes:
            label = shape.get("label", "").lower().strip()
            class_id = LABEL_MAP.get(label)
            if class_id is None:
                print(f"  Unknown label '{label}' in {json_path.name} — skipping shape")
                skipped += 1
                continue

            points = shape["points"]
            shape_type = shape.get("shape_type", "polygon")

            if shape_type == "rectangle":
                # points = [[x1,y1],[x2,y2]]
                x1, y1 = points[0]
                x2, y2 = points[1]
                x_min, y_min, x_max, y_max = min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)
            else:
                x_min, y_min, x_max, y_max = polygon_to_bbox(points)

            cx = ((x_min + x_max) / 2) / img_w
            cy = ((y_min + y_max) / 2) / img_h
            w  = (x_max - x_min) / img_w
            h  = (y_max - y_min) / img_h

            # Clamp to [0, 1]
            cx, cy, w, h = (max(0.0, min(1.0, v)) for v in (cx, cy, w, h))

            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        txt_path = json_path.with_suffix(".txt")
        txt_path.write_text("\n".join(lines))
        print(f"  {json_path.name} -> {txt_path.name}  ({len(lines)} box(es))")
        ok += 1

    print(f"\nDone: {ok} converted, {no_shapes} empty, {skipped} unknown labels skipped.")


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/golden_data")
    folder = folder.resolve()
    if not folder.exists():
        sys.exit(f"Folder not found: {folder}")
    convert(folder)


if __name__ == "__main__":
    main()

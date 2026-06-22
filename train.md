# Training a Custom YOLOv8 EV Charging Station Detector

**Goal:** A TF.js-compatible object detection model that recognises EV charging
stations in phone camera frames, so the AR app can snap arrows to the real
object instead of relying on GPS alone.

---

## Overview of the pipeline

```
Collect images
      ↓
Auto-label with SAM2 + Grounding DINO  (review & fix)
      ↓
Export dataset in YOLO format
      ↓
Fine-tune YOLOv8n on your data
      ↓
Evaluate  →  iterate
      ↓
Export → TF.js SavedModel
      ↓
Drop into index.html
```

---

## 1. Data collection

### How many images?
| Goal | Minimum images |
|---|---|
| Proof-of-concept (1 charger type) | 150–200 |
| Useful in the field (mixed chargers) | 500–800 |
| Production quality | 1 500+ |

### What to capture
- Multiple charger types: CCS2, CHAdeMO, Type 2 AC, Tesla V3, wallboxes
- Distance range: 3 m – 30 m  
- Angles: front-on, 45°, side
- Lighting: daylight, overcast, dusk, indoor parking, direct sun
- Occlusion: partially hidden by a car, person standing nearby
- Backgrounds: car park, street, garage, forecourt

### Where to get images
1. **Shoot yourself** — fastest path to domain-matched data; use the AR app's
   camera for realistic perspective/resolution
2. **Open Images v7** — Google's dataset has an `Electric vehicle charging station`
   class (~2 000 images, already annotated as bounding boxes):
   ```bash
   pip install fiftyone
   python - <<'EOF'
   import fiftyone.zoo as foz
   ds = foz.load_zoo_dataset(
       "open-images-v7",
       split="train",
       label_types=["detections"],
       classes=["Electric vehicle charging station"],
       max_samples=1000,
   )
   ds.export("./openimages_ev", dataset_type=foz.types.YOLOv5Dataset)
   EOF
   ```
3. **Roboflow Universe** — search `EV charging station`; several public datasets
   are freely downloadable in YOLO format
4. **Web scraping** — use a script + manual review; avoid copyright issues

---

## 2. Labeling — current SOTA

### The two-step SOTA workflow (2024–2025)
Modern labeling combines two foundation models:

| Model | Role |
|---|---|
| **Grounding DINO** | Zero-shot open-vocabulary *detection* — finds boxes from a text prompt like `"EV charging station"` |
| **SAM 2** (Meta, 2024) | Converts those boxes into precise *pixel masks* (or tight boxes) with a single click; also tracks across video frames |

In practice: Grounding DINO proposes bounding boxes → SAM 2 refines them →
human reviews and fixes edge cases. This reduces manual labeling effort by 70–90 %.

### Recommended tool: Roboflow (free tier)

Roboflow is the fastest end-to-end pipeline for a solo developer. It wraps
Grounding DINO + SAM 2 behind a browser UI called **"Label Assist"**.

1. Create a free account at roboflow.com  
2. New project → Object Detection → class name `charger`  
3. Upload images  
4. Open the labeler → click **Label Assist** → type `EV charging station`  
   Grounding DINO auto-draws boxes; SAM 2 tightens them  
5. Review every image — accept, adjust, or reject each box  
6. Export → **YOLOv8 format** (gives you `data.yaml` + `images/` + `labels/`)

Free tier: 10 000 source images / month, unlimited versions.

### Alternative: CVAT (self-hosted or cloud, open-source)

```bash
# Local install via Docker
git clone https://github.com/cvat-ai/cvat
cd cvat
docker compose up -d
```

CVAT ≥ 2.4 has a built-in **SAM 2 interactor** under
*Models → Interactors → Segment Anything 2*. Click on a charger → SAM 2 draws
the polygon → convert to a bounding box. Slower per-image than Roboflow's
auto-label but fully local / no data leaves your machine.

Export: *Tasks → Export dataset → YOLO 1.1* format.

### Alternative: Label Studio + ML backend

Good if you already have a running server. Plug in a Grounding DINO ML backend:

```bash
pip install label-studio label-studio-ml
git clone https://github.com/HumanSignal/label-studio-ml-backend
cd label-studio-ml-backend/label_studio_ml/examples/grounding_dino
pip install -r requirements.txt
label-studio-ml start . --port 9090
```

Then in Label Studio: *Settings → ML → Add Model → http://localhost:9090*.

---

## 3. Dataset structure

After export you should have:

```
dataset/
  data.yaml
  images/
    train/   # 80 %
    val/     # 15 %
    test/    # 5 %
  labels/
    train/
    val/
    test/
```

`data.yaml` (Roboflow generates this automatically):

```yaml
path: ./dataset
train: images/train
val:   images/val
test:  images/test

nc: 1
names: ['charger']
```

Each `.txt` label file (one per image) contains one row per box:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values normalised 0–1 relative to image dimensions.

---

## 4. Training YOLOv8

### Install

```bash
pip install ultralytics>=8.3
```

### Choose the right model variant

| Variant | Parameters | Inference (GPU) | Use case |
|---|---|---|---|
| `yolov8n` | 3.2 M | ~1 ms | **Mobile / browser — use this** |
| `yolov8s` | 11 M | ~2 ms | Desktop inference |
| `yolov8m` | 25 M | ~5 ms | Higher accuracy, still fast on GPU |

For TF.js in a phone browser, `yolov8n` is the only realistic choice.

### Fine-tune from the pretrained COCO checkpoint

```bash
yolo detect train \
  model=yolov8n.pt \
  data=dataset/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  name=ev_charger_v1 \
  patience=20
```

Key arguments:

| Argument | Meaning |
|---|---|
| `model=yolov8n.pt` | Start from COCO pretrained weights (transfer learning) |
| `epochs=100` | With ~500 images, 100 epochs is a good starting point |
| `imgsz=640` | Standard YOLO input resolution |
| `batch=16` | Reduce to 8 if you run out of GPU memory |
| `patience=20` | Early stop if val mAP doesn't improve for 20 epochs |

Results land in `runs/detect/ev_charger_v1/`:

```
runs/detect/ev_charger_v1/
  weights/
    best.pt    ← use this
    last.pt
  results.png
  confusion_matrix.png
  val_batch0_pred.jpg
```

### Data augmentation (Ultralytics defaults are already good)

The built-in augmentations (mosaic, mixup, random flip, HSV jitter, random
crop) are active by default. For outdoor charger images specifically, add
extra brightness/contrast variation to handle dawn/dusk conditions:

```python
# train.py (alternative to CLI)
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    hsv_h=0.02,   # hue jitter
    hsv_s=0.8,    # saturation jitter
    hsv_v=0.5,    # brightness jitter  ← key for lighting variation
    degrees=5,    # small rotation
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    name='ev_charger_v1',
)
```

---

## 5. Evaluation

After training, validate on the held-out test set:

```bash
yolo detect val \
  model=runs/detect/ev_charger_v1/weights/best.pt \
  data=dataset/data.yaml \
  split=test
```

Key metrics to check:

| Metric | Acceptable | Good |
|---|---|---|
| mAP50 | > 0.65 | > 0.80 |
| mAP50-95 | > 0.40 | > 0.60 |
| Recall | > 0.70 | > 0.85 |
| Precision | > 0.70 | > 0.85 |

**If mAP is low:**
- Add more diverse images of the failure cases (run `val_batch*_pred.jpg` to see misses)
- Check for label errors — mislabelled boxes are the #1 cause of poor recall
- Lower the confidence threshold at inference time (`conf=0.25`)

**Visualise predictions on your own photos:**

```bash
yolo detect predict \
  model=runs/detect/ev_charger_v1/weights/best.pt \
  source=my_test_photos/ \
  conf=0.3 \
  save=True
```

---

## 6. Export to TF.js

```bash
yolo export \
  model=runs/detect/ev_charger_v1/weights/best.pt \
  format=tfjs \
  imgsz=320      # use 320 for browser — half the compute of 640
```

This produces:

```
runs/detect/ev_charger_v1/weights/best_web_model/
  model.json
  group1-shard1of4.bin
  group1-shard2of4.bin
  ...
```

Host these files on any static server (GitHub Pages, S3, your own HTTPS host)
or bundle them alongside `index.html`.

> **CORS:** The model files must be served from the **same origin** as
> `index.html`, or the server must send `Access-Control-Allow-Origin: *`.
> GitHub Pages does this automatically.

---

## 7. Swap the model into the app

In `index.html`, find the marked swap block inside `startAR()` and replace the
`cocoSsd.load(...)` call:

```js
// BEFORE (COCO-SSD placeholder):
cocoSsd.load({ base: 'lite_mobilenet_v2' })
  .then(model => { ... })

// AFTER (your custom YOLOv8 TF.js model):
tf.loadGraphModel('./best_web_model/model.json')
  .then(rawModel => {
    // Wrap into a duck-typed object matching the cocoSsd API surface used
    // by runDetectionLoop()
    tfModel = makeYoloWrapper(rawModel, ['charger']);
    ...
  })
```

You also need to replace `tfModel.detect(canvas)` in `runDetectionLoop()` with
a YOLOv8-specific inference call, because YOLOv8 TF.js output tensors differ
from COCO-SSD's. A minimal wrapper:

```js
function makeYoloWrapper(graphModel, classNames) {
  return {
    async detect(canvas) {
      // Pre-process
      const input = tf.tidy(() => {
        return tf.image
          .resizeBilinear(tf.browser.fromPixels(canvas), [320, 320])
          .div(255.0)
          .expandDims(0);          // [1, 320, 320, 3]
      });

      // Run model — YOLOv8 TF.js output shape: [1, 5+nc, 8400]
      const raw = await graphModel.executeAsync(input);
      input.dispose();

      // raw[0] or raw itself depending on export version
      const tensor = Array.isArray(raw) ? raw[0] : raw;
      const data   = await tensor.data();
      tensor.dispose();

      // Decode — rows are [x, y, w, h, conf, cls0, cls1, ...]
      const numDetections = 8400;
      const numFields     = classNames.length + 4;  // 4 box coords
      const results = [];

      for (let i = 0; i < numDetections; i++) {
        let maxConf = 0, maxCls = 0;
        for (let c = 0; c < classNames.length; c++) {
          const conf = data[(4 + c) * numDetections + i];
          if (conf > maxConf) { maxConf = conf; maxCls = c; }
        }
        if (maxConf < 0.25) continue;

        const cx = data[0 * numDetections + i] / 320 * canvas.width;
        const cy = data[1 * numDetections + i] / 320 * canvas.height;
        const bw = data[2 * numDetections + i] / 320 * canvas.width;
        const bh = data[3 * numDetections + i] / 320 * canvas.height;

        results.push({
          class: classNames[maxCls],
          score: maxConf,
          bbox: [cx - bw / 2, cy - bh / 2, bw, bh],
        });
      }

      // Basic NMS — keep highest confidence when boxes overlap heavily
      return nms(results, 0.45);
    }
  };
}

function nms(boxes, iouThreshold) {
  boxes.sort((a, b) => b.score - a.score);
  const kept = [];
  const suppressed = new Set();
  for (let i = 0; i < boxes.length; i++) {
    if (suppressed.has(i)) continue;
    kept.push(boxes[i]);
    for (let j = i + 1; j < boxes.length; j++) {
      if (iou(boxes[i].bbox, boxes[j].bbox) > iouThreshold) suppressed.add(j);
    }
  }
  return kept;
}

function iou([x1,y1,w1,h1], [x2,y2,w2,h2]) {
  const ix = Math.max(0, Math.min(x1+w1, x2+w2) - Math.max(x1, x2));
  const iy = Math.max(0, Math.min(y1+h1, y2+h2) - Math.max(y1, y2));
  const inter = ix * iy;
  return inter / (w1*h1 + w2*h2 - inter + 1e-6);
}
```

---

## 8. Iteration loop

```
deploy → test in field → note failure cases
→ collect 50–100 more images of the failure cases
→ label (Roboflow auto-label + review)
→ re-train (just add new images to existing dataset — no need to start over)
→ re-export → redeploy
```

Three or four of these rounds typically bring a `yolov8n` model from ~0.65 to
~0.85 mAP50 on real-world charging station imagery.

---

## Quick-reference commands

```bash
# Install
pip install ultralytics>=8.3 fiftyone

# Download Open Images EV data (optional starting point)
python collect_openimages.py

# Train
yolo detect train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640

# Validate
yolo detect val model=runs/detect/ev_charger_v1/weights/best.pt data=dataset/data.yaml split=test

# Test on photos
yolo detect predict model=runs/detect/ev_charger_v1/weights/best.pt source=photos/ conf=0.3

# Export to TF.js (use imgsz=320 for browser performance)
yolo export model=runs/detect/ev_charger_v1/weights/best.pt format=tfjs imgsz=320
```

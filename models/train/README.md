# Stream B — YOLO model

Fine-tune YOLOv8n to the 6 DC-Ops classes, then export to ExecuTorch `.pte`.

## 1. Get data
Lay out labeled images in YOLO format under `data/` (see `data/README.md`):
```
data/images/train/*.jpg   data/labels/train/*.txt
data/images/val/*.jpg     data/labels/val/*.txt
```
Classes are frozen in `models/classes.yaml` — train to those exact IDs.

## 2. Train
```bash
pip install ultralytics
python models/train/train_yolo.py --epochs 100 --imgsz 640
# → runs/detect/dcops_yolov8n/weights/best.pt
```

## 3. Export to .pte (Snapdragon NPU)
```bash
source scripts/setup_env.sh          # needs QNN_SDK_ROOT
python models/export/export_yolo.py --soc_model SM8750 \
    --weights runs/detect/dcops_yolov8n/weights/best.pt
```

## 4. Hand off (ARCHITECTURE.md §5.4)
- Filename: `yolov8n_dcops_int8.pte`
- Drop it in `android-app/app/src/main/assets/models/` (or attach to a GitHub Release)
- **Document the output tensor spec** so Stream A can decode it into `DetectionResult`.

Until step 4 lands, the app runs on `StubModelManager` — nothing is blocked.

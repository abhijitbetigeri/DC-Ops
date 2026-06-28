"""Upload DC-Ops dataset + trained model to Hugging Face."""

from huggingface_hub import HfApi, create_repo
from pathlib import Path

REPO_ID = "abhijitbetigeri/dc-ops-dataset"

api = HfApi()

# Create the dataset repo
try:
    create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
    print(f"Repo ready: https://huggingface.co/datasets/{REPO_ID}")
except Exception as e:
    print(f"Repo creation: {e}")

# Upload images
print("\nUploading images...")
api.upload_folder(
    folder_path="data/sample_images",
    path_in_repo="images",
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Add 319 DC infrastructure images",
)

# Upload labels
print("Uploading labels...")
api.upload_folder(
    folder_path="data/labels_colab",
    path_in_repo="labels",
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Add YOLO-seg polygon labels (16 DC classes, 3045 objects)",
)

# Upload trained model
print("Uploading trained model...")
api.upload_file(
    path_or_fileobj="models/dc_ops_yolov8n_seg.pt",
    path_in_repo="models/dc_ops_yolov8n_seg.pt",
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Add YOLOv8n-seg trained weights (6.5MB)",
)

# Upload README
readme = """---
license: mit
task_categories:
  - object-detection
  - image-segmentation
tags:
  - data-center
  - server-rack
  - infrastructure
  - executorch
  - qualcomm
  - on-device-ai
  - yolov8
pretty_name: DC-Ops Data Center Components Dataset
size_categories:
  - n<1K
---

# DC-Ops: Data Center Components Dataset

On-device data center operations assistant dataset for the **Qualcomm x Meta ExecuTorch Hackathon**.

## Overview

- **319 images** of data center infrastructure (server racks, NVIDIA NVL72, cables, ports, etc.)
- **3,045 polygon annotations** in YOLO-seg format
- **16 component classes** auto-labeled with Grounding DINO + SAM, for fine-tuning YOLOv8n-seg

## Classes

| ID | Class | Count |
|---|---|---|
| 0 | server rack | rack enclosures |
| 1 | compute tray | GPU/CPU trays (NVL72: 18 per rack) |
| 2 | NVLink switch tray | NVLink switches |
| 3 | network switch | TOR / Ethernet / InfiniBand |
| 4 | power shelf | PSU banks |
| 5 | cable | copper, fiber, power |
| 6 | network port | OSFP, QSFP, Ethernet |
| 7 | LED indicator | status LEDs |
| 8 | label | serial numbers, asset tags |
| 9 | fan | cooling fans |
| 10 | cooling manifold | liquid cooling pipes |
| 11 | cable cartridge | NVLink cable cartridge |
| 12 | power connector | power plugs, bus bar |
| 13 | drive bay | NVMe drive slots |
| 14 | management port | BMC RJ45, serial console |
| 15 | DPU | BlueField DPU / ConnectX NIC |

## Trained Model

`models/dc_ops_yolov8n_seg.pt` — YOLOv8n-seg fine-tuned on this dataset (6.5MB)

- Box mAP50: 0.232
- Top classes: compute tray (0.57), network port (0.51), cable (0.27)
- Target device: Samsung Galaxy S25 Ultra (Snapdragon 8 Elite, SM8750)

## Label Format

YOLO-seg format: `class_id x1 y1 x2 y2 ... xn yn` (normalized polygon coordinates)

## Pipeline

```
Brightdata scraping → Grounding DINO + SAM auto-labeling → YOLOv8n-seg fine-tuning → ExecuTorch .pte export → Snapdragon NPU
```

## Usage

```python
from ultralytics import YOLO
model = YOLO("models/dc_ops_yolov8n_seg.pt")
results = model("path/to/server_rack_image.jpg")
```
"""

api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Add dataset card",
)

print(f"\nDone! https://huggingface.co/datasets/{REPO_ID}")

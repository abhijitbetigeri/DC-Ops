# DC-Ops: On-Device Data Center Operations Assistant

**Qualcomm x Meta ExecuTorch Hackathon — June 27–28, 2026**

Real-time camera-based tool for data center technicians. Point your phone at server racks to detect components — compute trays, network ports, LEDs, cables, fans, drive bays, and more — with on-device AI. Includes an intelligent RAG-powered knowledge overlay for troubleshooting, specs, and maintenance guidance.

## Why On-Device?

- **Air-gapped environments**: Many data centers prohibit cloud connectivity in server rooms
- **Privacy**: Serial numbers, rack topology, and infrastructure status are sensitive/classified
- **Latency**: Technicians need instant visual feedback walking through aisles
- **Offline-first**: Even connected DCs have dead zones between rack rows

## Architecture

```
Camera (CameraX, 30fps)
 │
 ▼
RetinaNet-ResNet50-FPN (INT8 quantized)
 ├─ ExecuTorch → QNN HTP → Snapdragon NPU (primary, ~3-5ms)
 └─ ExecuTorch → XNNPACK → CPU (fallback, ~15-30ms)
 │
 ├─ 16 DC component classes detected with bounding polygons
 ├─ Compute trays, NVLink switches, network ports, LEDs, cables, etc.
 │
 ▼
RAG Knowledge Overlay (CLIP ViT-B-32 + FAISS)
 ├─ Detected component → text embedding → nearest-neighbor search
 ├─ Returns: specs, troubleshooting, LED meanings, maintenance steps
 └─ 80 knowledge chunks, 160KB index, fully on-device
 │
 ▼
AR Overlay + Live Metrics
 ├─ Color-coded polygon overlays on camera feed (16 distinct colors)
 ├─ Live FPS + latency display
 ├─ CPU ↔ NPU backend toggle for comparison
 └─ Tap component → info panel with specs + troubleshooting
```

## Models

| Model | Backend | Size | Purpose | Status |
|---|---|---|---|---|
| RetinaNet-ResNet50-FPN | **QNN HTP (NPU)** | 36 MB | Primary detection on Snapdragon NPU | ✅ Ready |
| YOLOv8n-seg v3 | XNNPACK (CPU) | 13 MB | CPU fallback + comparison | ✅ Ready |
| CLIP ViT-B-32 | CPU | 160 KB index | RAG knowledge retrieval | ✅ Ready |

## 16 DC Component Classes

| ID | Class | What it detects |
|---|---|---|
| 0 | server rack | Full rack enclosure |
| 1 | compute tray | GPU/CPU trays (NVL72: 18 per rack) |
| 2 | NVLink switch tray | NVLink switches (NVL72: 9 per rack) |
| 3 | network switch | TOR / Ethernet / InfiniBand switches |
| 4 | power shelf | PSU banks (NVL72: 8 per rack, 6x 5.5kW each) |
| 5 | cable | Copper, fiber, power cables |
| 6 | network port | OSFP, QSFP, Ethernet, USB, HDMI ports |
| 7 | LED indicator | Status LEDs on trays/switches |
| 8 | label | Serial numbers, asset tags |
| 9 | fan | Cooling fans |
| 10 | cooling manifold | Liquid cooling pipes/connections |
| 11 | cable cartridge | NVLink copper cable cartridge |
| 12 | power connector | Power plugs, bus bar connectors |
| 13 | drive bay | NVMe drive slots |
| 14 | management port | BMC RJ45, serial console |
| 15 | DPU | BlueField DPU / ConnectX NIC |

## Training Data

**2,036 human-labeled images** from 4 Roboflow datasets (CC BY 4.0):

| Dataset | Images | Source Classes |
|---|---|---|
| [Server Vision](https://universe.roboflow.com/hunters-workspace-kqclz/server-vision) | 1,293 | 30 (CPU sockets, DIMM slots, fans, PSUs, drives, ports) |
| [PC Ports](https://universe.roboflow.com/ports-gmmmp/pc-ports) | 327 | 4 (HDMI, RJ45, USB-A, USB-C) |
| [Ports](https://universe.roboflow.com/ym-pffnw/ports-yutnj) | 138 | 7 (DP, ethernet, power, USB variants) |
| [Server Detection](https://universe.roboflow.com/fujitsu-ih8ei/server-detection) | 54 | 9 (SFP, XFP, patch panels, servers) |

All source classes mapped to our 16 DC-Ops classes via `scripts/merge_datasets.py`.

## RAG Knowledge Base

On-device retrieval-augmented generation for component information:

- **80 knowledge chunks** covering all 16 component classes
- Each chunk contains: description, specs, troubleshooting, LED status meanings, maintenance procedures
- **NVIDIA GB200 NVL72** specific documentation included
- CLIP ViT-B-32 text embeddings + FAISS nearest-neighbor search
- **160 KB** total index size — fully on-device, no cloud needed

Example: detect "compute tray" → RAG returns:
> *"1RU compute tray: 2x Grace CPUs, 4x B200 GPUs, 192GB HBM3e. Amber LED = check BMC log. Verify coolant flow rate."*

## Tech Stack

| Component | Tool | Runtime |
|---|---|---|
| Detection | RetinaNet-ResNet50-FPN-v2 (INT8) | ExecuTorch → QNN HTP (Snapdragon NPU) |
| Detection (fallback) | YOLOv8n-seg v3 | ExecuTorch → XNNPACK (CPU) |
| RAG retrieval | CLIP ViT-B-32 + FAISS | On-device |
| Android app | Kotlin + CameraX + ExecuTorch AAR | Galaxy S25 Ultra |
| Model format | .pte (ExecuTorch) | Compiled for SM8750 |
| Knowledge base | JSON + FAISS index | 160 KB on-device |

## Target Device

- **Samsung Galaxy S25 Ultra**
- **Chipset**: Snapdragon 8 Elite for Galaxy (SM8750-AC)
- **Hexagon**: v79 / SOC Model 69
- **NPU**: Hexagon Tensor Processor (HTP) for accelerated inference

## Project Structure

```
DC-Ops/
├── android-app/              # Android application
│   └── app/
│       ├── libs/             # executorch.aar (3.4 MB)
│       └── src/main/
│           ├── assets/       # .pte models (QNN + XNNPACK)
│           └── java/com/dcops/ar/
│               ├── MainActivity.kt          # Camera + backend toggle + metrics
│               ├── camera/CameraManager.kt  # CameraX preview + analysis
│               ├── inference/
│               │   ├── ModelManager.kt      # ExecuTorch inference + dual backend
│               │   └── DetectionResult.kt   # Detection data class
│               └── overlay/
│                   └── PolygonOverlayView.kt # AR polygon overlay (16 colors)
├── data/
│   ├── knowledge_base/       # RAG index + component documentation
│   ├── sample_images/        # Training images
│   └── merged_dataset/       # Merged Roboflow datasets (YOLO format)
├── models/                   # Trained weights + .pte files
├── notebooks/                # Colab notebooks for training + export
│   ├── train_merged_colab.ipynb          # Train on 2,036 Roboflow images
│   ├── train_retinanet_qnn_colab.ipynb   # RetinaNet + QNN HTP export
│   ├── export_qnn_colab.ipynb            # QNN export standalone
│   └── build_rag_colab.ipynb             # RAG index builder
├── scripts/                  # Build, train, and utility scripts
│   ├── merge_datasets.py     # Merge 4 Roboflow datasets
│   ├── build_rag_index.py    # CLIP embeddings + FAISS index
│   ├── benchmark_on_device.sh # CPU vs NPU benchmark
│   └── setup_env.sh          # Environment setup
├── executorch/               # ExecuTorch SDK (gitignored)
├── CLAUDE.md
└── README.md
```

## Setup

### Prerequisites
- Python 3.12
- JDK 17
- Android NDK v26
- QAIRT SDK 2.46.0

### Quick Start
```bash
source scripts/setup_env.sh
```

### Build Android App
```bash
cd android-app
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

### Train Models (Colab)
Upload notebooks to Google Colab (T4 GPU):
1. `train_retinanet_qnn_colab.ipynb` — fine-tune RetinaNet + QNN HTP export
2. `train_merged_colab.ipynb` — train on merged Roboflow datasets

### Build RAG Index
```bash
python scripts/build_rag_index.py
```

## Demo Plan (5 min)

1. **(0:00–1:00)** Point phone at server rack → real-time component detection with colored overlays
2. **(1:00–2:00)** Tap a detected component → RAG info panel shows specs + troubleshooting
3. **(2:00–3:00)** Toggle CPU ↔ NPU → show FPS/latency difference (15ms vs 3ms)
4. **(3:00–4:00)** Enable airplane mode → everything still works (air-gapped demo)
5. **(4:00–5:00)** Architecture walkthrough: ExecuTorch + QNN HTP + on-device RAG

## Models on Hugging Face

All models available at: [abhijitbetigeri/dc-ops-dataset](https://huggingface.co/datasets/abhijitbetigeri/dc-ops-dataset)

| File | Description |
|---|---|
| `models/dc_ops_retinanet_qnn.pte` | RetinaNet QNN HTP for Snapdragon NPU (36 MB) |
| `models/dc_ops_yolov8n_seg_v3.pte` | YOLOv8n-seg XNNPACK for CPU (13 MB) |
| `models/dc_ops_yolov8n_seg_v3.pt` | YOLOv8n-seg PyTorch weights (6.4 MB) |

## Performance

### RetinaNet (trained on 2,036 images, 10 epochs)
- Final training loss: **1.61**
- Backend: QNN HTP (INT8 quantized) for SM8750

### YOLOv8n-seg v3 (trained on 2,036 images, 50 epochs)
| Class | mAP50 |
|---|---|
| compute tray | **0.924** |
| network port | **0.912** |
| power shelf | **0.832** |
| LED indicator | **0.808** |
| server rack | 0.683 |
| cable | 0.660 |
| **Overall** | **0.749** |

## Team

| Role | Focus |
|---|---|
| Model Engineer | RetinaNet + QNN HTP export, data pipeline, RAG |
| Android Developer | CameraX, ExecuTorch AAR, AR overlay, backend toggle |
| Data + Demo | Roboflow datasets, demo scenario, presentation |

# DC-Ops

On-Device Data Center Operations Assistant for the Qualcomm x Meta ExecuTorch Hackathon.

## Project

Real-time camera-based tool for data center technicians. Detects 16 types of DC components with AR overlay, plus RAG-powered knowledge retrieval for specs and troubleshooting — all on-device on Samsung Galaxy S25 Ultra (Snapdragon 8 Elite, SM8750).

## Architecture

- **Detection (primary)**: RetinaNet-ResNet50-FPN (INT8) → ExecuTorch → QNN HTP → Snapdragon NPU
- **Detection (fallback)**: YOLOv8n-seg v3 → ExecuTorch → XNNPACK → CPU
- **RAG**: CLIP ViT-B-32 embeddings + FAISS index (160KB, on-device)
- **App**: Android (Kotlin) + CameraX + ExecuTorch AAR + backend toggle

## Key Paths

- `android-app/` — Android application (Kotlin, CameraX, ExecuTorch)
- `data/knowledge_base/` — RAG index + component documentation
- `models/` — Trained weights + .pte files
- `notebooks/` — Colab notebooks for training + QNN export
- `scripts/` — Merge datasets, build RAG, benchmark
- `executorch/` — ExecuTorch SDK (gitignored)

## Models

- `dc_ops_retinanet_qnn.pte` — RetinaNet QNN HTP for NPU (36MB)
- `dc_ops_yolov8n_seg.pte` — YOLOv8n-seg XNNPACK for CPU (13MB)

## Environment

```bash
source scripts/setup_env.sh
```

## Build

```bash
cd android-app && ./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Target Device

Samsung Galaxy S25 Ultra — Snapdragon 8 Elite (SM8750-AC), Hexagon v79

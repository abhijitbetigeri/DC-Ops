# DC-Ops: On-Device Data Center Operations Assistant

**Qualcomm x Meta ExecuTorch Hackathon — June 27–28, 2026**

Real-time camera-based tool for data center technicians. Point your phone at server racks to detect status LEDs, read serial numbers via OCR, and flag cable issues — all on-device, all offline.

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
YOLOv8n (quantized INT8, ~6MB)     ──► ExecuTorch → Snapdragon NPU (QNN)
 ├─ LED detection + color classification (green / amber / red / off)
 ├─ Cable presence / absence detection
 └─ Label / nameplate region detection
 │
 ▼
PP-OCRv3 mobile (~5MB)              ──► ExecuTorch → Snapdragon NPU (QNN)
 └─ Serial number / asset tag extraction from detected label regions
 │
 ▼
AR Overlay + Local Audit Log
 ├─ Bounding boxes with status annotations on live camera feed
 └─ SQLite log (timestamped findings, exportable via secure channel)
```

## Tech Stack

| Component | Model / Tool | Runtime |
|---|---|---|
| Object detection | YOLOv8n (INT8 quantized) | ExecuTorch → Snapdragon NPU (QNN backend) |
| OCR | PP-OCRv3 mobile or CRNN | ExecuTorch → Snapdragon NPU (QNN backend) |
| Android UI | Kotlin + CameraX + ExecuTorch AAR | Galaxy S25 Ultra |
| Local storage | SQLite | On-device |
| Model format | .pte (ExecuTorch) | Compiled for SM8750 (Snapdragon 8 Elite) |

## Target Device

- **Samsung Galaxy S25 Ultra**
- **Chipset**: Snapdragon 8 Elite for Galaxy (SM8750-AC)
- **Hexagon**: v79 / SOC Model 69

## Project Structure

```
DC-Ops/
├── models/
│   └── export/          # PyTorch → ExecuTorch export scripts
├── android/
│   └── app/             # Android app (Kotlin, CameraX, ExecuTorch runtime)
├── data/
│   └── sample_images/   # Training/test images of server racks
├── scripts/             # Build, deploy, and utility scripts
├── executorch/          # ExecuTorch SDK (cloned, gitignored)
└── README.md
```

## Setup

### Prerequisites
- Python 3.12
- JDK 17
- Android NDK v26
- QAIRT SDK 2.46.0.260424

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export ANDROID_HOME=/opt/homebrew/share/android-commandlinetools
export ANDROID_NDK_ROOT=$ANDROID_HOME/ndk/26.3.11579264
export QNN_SDK_ROOT=<path_to_qairt_sdk>/2.46.0.260424
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/<toolchain>:$LD_LIBRARY_PATH
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
```

### Model Export

> ⚠️ **Do not do a naive full-model export.** The YOLOv8-seg head won't lower to QNN and the
> export silently falls back to a **CPU build** (looks correct, never touches the NPU). You must
> export the **backbone only**. See **[`MODEL_BUILD_GUIDE.md`](MODEL_BUILD_GUIDE.md)** for the
> verified, step-by-step recipe (and the size / `QnnBackend` check that tells a good build from a
> bad one).

```bash
source scripts/qnn/env.sh
MODEL_PT=<your.pt> OUT_PTE_BASE=<out> python scripts/qnn/export_yolo_backbone_qnn.py
# then VERIFY: ~3.8 MB and `strings <pte> | grep -c QnnBackend` >= 1  (13 MB / 0 = CPU build = wrong)
```

## Demo Plan (5 min)

1. **(0:00–1:00)** Point phone at server rack → real-time LED status overlay (green ✓, red ✗)
2. **(1:00–2:00)** Tap a server → OCR reads serial/asset tag, logs it locally
3. **(2:00–3:00)** Show disconnected cable → app flags with warning annotation
4. **(3:00–4:00)** Enable airplane mode → everything still works perfectly
5. **(4:00–5:00)** Show local audit log + architecture walkthrough + latency numbers

## Team

| Role | Focus |
|---|---|
| Model Engineer | YOLOv8n + OCR → quantize → QNN backend → .pte files |
| Android Developer | CameraX, ExecuTorch AAR integration, AR overlay UI |
| Data + Demo | Training data curation, demo scenario, presentation |

# DC-Ops

On-Device Data Center Operations Assistant for the Qualcomm x Meta ExecuTorch Hackathon.

## Project

Real-time camera-based tool for data center technicians. Detects LED status, reads serial numbers via OCR, flags cable issues — all on-device on Samsung Galaxy S25 Ultra (Snapdragon 8 Elite, SM8750).

## Architecture

- **Detection**: YOLOv8n (INT8 quantized) → ExecuTorch → QNN → Snapdragon NPU
- **OCR**: Lightweight CRNN → ExecuTorch → QNN → Snapdragon NPU
- **App**: Android (Kotlin) + CameraX + ExecuTorch AAR
- **Storage**: Local SQLite audit log

## Key Paths

- `models/export/` — PyTorch → .pte export scripts
- `android/app/` — Android application
- `scripts/` — Build and setup utilities
- `executorch/` — ExecuTorch SDK (gitignored)

## Environment

```bash
source scripts/setup_env.sh
```

## Build

> ⚠️ Building a QNN `.pte` that actually runs on the NPU: see **`MODEL_BUILD_GUIDE.md`**.
> A naive full-model export silently produces a **CPU build** — you must export the backbone only,
> then verify (~3.8 MB, `strings <pte> | grep -c QnnBackend` >= 1). Full postmortem in
> `EXECUTORCH_QNN_NPU_NOTES.md`.

```bash
source scripts/qnn/env.sh
# Build the model (backbone-only QNN export):
MODEL_PT=<your.pt> OUT_PTE_BASE=<out> python scripts/qnn/export_yolo_backbone_qnn.py
# Deploy a model bundle to the device (no app/server rebuild needed):
scripts/qnn/swap_model.sh models/<name>
```

## Target Device

Samsung Galaxy S25 Ultra — Snapdragon 8 Elite (SM8750-AC), Hexagon v79

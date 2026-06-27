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

```bash
# Export models
python models/export/export_yolo.py --soc_model SM8750
python models/export/export_ocr.py --soc_model SM8750

# Build Android runtime
cd executorch && ./scripts/build_android_library.sh
```

## Target Device

Samsung Galaxy S25 Ultra — Snapdragon 8 Elite (SM8750-AC), Hexagon v79

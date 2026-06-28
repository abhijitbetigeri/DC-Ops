# How to build a DC-Ops QNN model `.pte` that actually runs on the NPU

This is the step-by-step recipe for turning a YOLOv8-seg `.pt` into a `.pte` that runs on the
Galaxy S25 Ultra's Hexagon NPU. Read the **one critical gotcha** first — it's the thing that
silently bites everyone.

> Deep background (every error we hit and why) lives in `EXECUTORCH_QNN_NPU_NOTES.md`.
> This file is just "do these steps and verify."

---

## ⚠️ The one critical gotcha — DON'T export the whole model

The "obvious" approach — export the full YOLOv8-seg model to QNN — **does not work**. The
detection/segmentation *head* uses ops the QNN backend can't quantize/compile, so the export
**falls back to a CPU build**. You get a `.pte` that loads fine and even produces detections...
on the CPU. It never touches the NPU.

**This is how the broken model gets made.** A plain `.pt → .pte` export (including the one that was
sitting in the HuggingFace `models/` folder, `dc_ops_yolov8n_seg_v3.pt` → 13 MB `.pte`) is a
**CPU export**. It looks correct. It is not.

### A good build vs. a bad build (always check this)

| | ✅ Correct (NPU) | ❌ Broken (CPU) |
|---|---|---|
| Size | **~3.8 MB** | ~13 MB |
| `strings model.pte \| grep -c QnnBackend` | **≥ 1** | **0** |
| Runs on | Hexagon v79 NPU (~3 ms/frame) | CPU (slow) |

**The fix is to export the BACKBONE ONLY** (the raw pre-decode conv outputs) and do the YOLO
decode (DFL, anchors, NMS, mask) on the other side. That's what `export_yolo_backbone_qnn.py`
does. Don't write your own naive exporter.

---

## Prerequisites — a Linux x86_64 build box

> **First time on a fresh machine?** Run the automated setup, then come back here:
> **[`setup/`](setup/README.md)** — `bash setup/setup.sh` (Linux), `./setup/setup.ps1` (Windows →
> WSL2), or `bash setup/run-in-docker.sh` (macOS → Linux container). It installs everything below.

The build host **must be Linux x86_64** (the QNN SDK ships only `x86_64-linux-clang` libs). Once
set up you have:

- conda env `executorch` (ExecuTorch `main` ≈ 1.4.0a0, `ultralytics` installed)
- QAIRT/QNN SDK **2.46.0.260424** at `~/et_setup/qairt/2.46.0.260424`
- Android NDK **r26c** at `~/et_setup/ndk/android-ndk-r26c`
- g++ **≥ 13** (the env script symlinks conda's gcc/g++ 13 — Ubuntu 22.04's 11 is too old)

All of this is wired up by **`~/et_setup/env.sh`** (a copy is `scripts/qnn/env.sh`). Just source it.

---

## Build steps

### 1. Source the environment
```bash
source ~/et_setup/env.sh        # (repo copy: scripts/qnn/env.sh)
# prints: QNN_SDK_ROOT, ANDROID_NDK_ROOT, g++ version
```

### 2. Export the backbone-only QNN `.pte`
The exporter is **`scripts/qnn/export_yolo_backbone_qnn.py`** (lives in `~/et_setup/` on the build
box). It takes two env vars:

```bash
# MODEL_PT     = path to your source .pt  (omit to use the cached/default base model)
# OUT_PTE_BASE = output path WITHOUT the .pte extension

MODEL_PT=/path/to/dc_ops_yolov8n_seg_v3.pt \
OUT_PTE_BASE=$HOME/executorch/dcops_qnn/dc_ops_yolov8n_seg_v3_backbone_qnn \
python ~/et_setup/export_yolo_backbone_qnn.py
```

What it does, and what to watch for in the output:
- Loads the `.pt`, wraps it so `forward()` returns **only** the 5 raw outputs
  (`ps0, ps1, ps2, mask_coeffs, proto`) — the un-lowerable head is dropped and DCE removes it.
- Prints `CLASS_NAMES: [...]` ← **copy these into `model.json`** so labels match exactly.
- Prints `raw outputs: [...]` ← sanity-check the shapes, e.g.
  `(1,80,80,80)(1,80,40,40)(1,80,20,20)(1,32,8400)(1,32,160,160)`.
- Prints `decode ops still present (want []): []` ← must be **empty**; if not, the head wasn't
  fully dropped.
- Downloads ~200 real DC images for INT8 **w8a16** PTQ calibration (cached in
  `~/et_setup/calib_imgs` so re-runs are fast). Quantization is `QuantDtype.use_16a8w` — YOLO's
  recommended HTP precision; `8a8w` collapsed scores to zero detections.
- Compiles for **`SM8750`** with `compile_only=True`.
- Prints `WROTE <path>.pte <size> bytes`.

### 3. VERIFY it's a real NPU build (do not skip)
```bash
PTE=$HOME/executorch/dcops_qnn/dc_ops_yolov8n_seg_v3_backbone_qnn.pte
ls -l "$PTE"                              # expect ~3.8 MB, NOT ~13 MB
strings "$PTE" | grep -c QnnBackend      # expect >= 1, NOT 0
```
If size is ~13 MB or the `QnnBackend` count is 0, it's a CPU build — **stop and fix the export**;
do not ship it.

### 4. Make a model bundle
A deployable model is a **bundle** = `model.pte` + `model.json`. Create a folder under `models/`:

```
models/dcops-v3/
├── model.pte      # the verified QNN .pte from step 2/3
└── model.json
```

`model.json` (the human-authored source of truth — see `models/dcops-v3/model.json`):
```json
{
  "type": "yolov8_seg",
  "input_size": 640,
  "reg_max": 16,
  "source": "dc_ops_yolov8n_seg_v3.pt, QNN backbone export w8a16, 200-img calib",
  "labels": ["server rack", "compute tray", "...16 names from CLASS_NAMES..."]
}
```
> Only `type`, `input_size`, `reg_max`, and `labels` are read. **`num_classes`, masks, mask
> resolution, grid scales, strides and anchor count are all DERIVED from the `.pte`'s tensor shapes
> at load** — you don't (and shouldn't) hand-write them. `labels` order must match the model's
> class indices (the `CLASS_NAMES:` line from step 2).

### 5. Deploy to the device
```bash
# Generates meta.txt + labels.txt from model.json, pushes the bundle to
# /data/local/tmp/dcops_yolo_qnn, restarts the server, and prints the
# derived-shapes / warmup lines so you can confirm it loaded correctly.
~/et_setup/swap_model.sh /mnt/c/Users/Rashi/AndroidStudioProjects/DC-Ops/models/dcops-v3
```
Confirm the server log shows your model (classes/input size match expectations), e.g.:
```
meta: type=yolov8_seg reg_max=16 input_size(meta)=640 labels=16
warmup forward: ok, outputs=5
model: type=yolov8_seg input=640x640 classes=16 masks=32 mask_res=160 scales=80/8,40/16,20/32 anchors=8400
```

> First time on a fresh device, the server binary + QNN libs must be staged once with
> `~/et_setup/push_persistent.sh` (swap_model.sh warns if it's missing). The server runs as the
> **shell** uid — see the NPU-in-app caveat in `EXECUTORCH_QNN_NPU_NOTES.md` for why it can't live
> in the APK.

---

## TL;DR for a coworker

1. `source scripts/qnn/env.sh`
2. `MODEL_PT=<your.pt> OUT_PTE_BASE=<out> python scripts/qnn/export_yolo_backbone_qnn.py`
3. **Verify**: `~3.8 MB` and `strings <pte> | grep -c QnnBackend` ≥ 1. (13 MB / 0 = CPU build = wrong.)
4. Put `model.pte` + `model.json` (labels from the `CLASS_NAMES:` line) in `models/<name>/`.
5. `scripts/qnn/swap_model.sh models/<name>` — no app or server rebuild needed for any YOLOv8-seg
   variant.

**Never** do a plain full-model `.pt → .pte` export and trust it — that silently produces a CPU
build (the trap that wasted a debugging cycle on the v3 model).

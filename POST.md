# DC-Ops: Running Real-Time Data-Center Inspection on a Phone's NPU

**Built at the Qualcomm × Meta ExecuTorch Hackathon**

## The problem we picked

Over **1,000 server racks are assembled every week**, by hand, at volume — crews wiring dense meshes of cables while cross-referencing paper guides. The industry reality: **~80% of racks are wired incorrectly on the first pass.** Wrong port, wrong cable, missed component.

So we built **DC-Ops**: point a phone at a server rack and it identifies the components in real time — compute trays, network ports, LEDs, cables, fans, drive bays, power shelves, DPUs — and guides the technician to the **exact port for the next cable.**

The catch that makes it interesting: **it all runs on the device.** No cloud, no network. Because data centers are air-gapped, infrastructure data is sensitive, and a technician walking the aisles needs instant, offline feedback.

## What we built

A native Android app on the **Samsung Galaxy S25 Ultra (Snapdragon 8 Elite)** with two modes:

- **Server Scan** — live multi-class detection with color-coded overlays
- **Cable Match ("Mission Control")** — an AR workflow that highlights the precise port for the next cable in the assembly sequence

Plus an **on-device RAG knowledge layer**: tap any detected component and it surfaces specs, LED-status meanings, and troubleshooting steps — using CLIP embeddings + a FAISS index that's just **160 KB**, bundled in the app. Zero network calls.

## The model

We fine-tuned **YOLOv8n-seg** for 16 data-center component classes. Getting training data for a niche domain like this was its own puzzle:

- **Bootstrapped** with a **BrightData** web-scraping pipeline + auto-labeling using Grounding DINO + SAM
- **Refined** on **2,036 human-labeled images** across 4 Roboflow datasets
- **Augmented** for moiré patterns, screen glare, and brightness — because half our demos are pointing a phone at a server *photo on a laptop screen*

Result: **0.749 mAP50**, with the components that matter most for cable-matching — ports and trays — detecting above **90%**.

## The hard part: actually using the NPU

Here's where it got real. "Runs on Snapdragon" usually means "runs on Snapdragon's CPU." We wanted it on the **Hexagon Tensor Processor** — the dedicated AI silicon — via **ExecuTorch + Qualcomm's QNN HTP backend.**

That meant fighting through, in order:
- Missing `libc++` in the build environment
- Dynamic-linker paths the QNN backend couldn't resolve
- `InitBackend` failures (fixed by running the export in a subprocess with the library path set *before* Python starts)
- And finally the real blocker: **YOLOv8's anchor-decode head uses dynamic ops** (`meshgrid`, `full`, `item`) that the NPU compiler simply can't lower.

The fix: **export only the raw convolution feature maps** — pure convs that compile cleanly to the Hexagon HTP — and move YOLO's decode (DFL → anchors → boxes → NMS) **into the app**, written in Kotlin.

The payoff:

| | CPU (XNNPACK) | **NPU (QNN HTP)** |
|---|---|---|
| Hardware | ARM cores | Hexagon Tensor Processor |
| Precision | FP32 | INT8 |
| Size | 13 MB | **3.2 MB** |

Same model, 4× smaller, running on hardware built for exactly this — and **both paths are selectable live in the app**, so you can watch CPU vs NPU side by side.

## The takeaway

The whole point of on-device AI isn't just "no cloud" — it's matching the *right workload* to the *right silicon*. We took a model from PyTorch → ExecuTorch → INT8 → the Hexagon NPU, and built a tool that turns a phone into a data-center technician's co-pilot that works in a room with no signal at all.

## Stack

PyTorch · **ExecuTorch** · **Qualcomm QNN HTP / Snapdragon 8 Elite** · YOLOv8 · CLIP + FAISS · Grounding DINO + SAM · **Roboflow** · **BrightData** · Kotlin + CameraX · Hugging Face · Three.js (interactive NVL72 rack model)

🔗 Code: https://github.com/abhijitbetigeri/DC-Ops
🤗 Models + dataset: https://huggingface.co/datasets/abhijitbetigeri/dc-ops-dataset

Huge thanks to Qualcomm and Meta for the hardware and the ExecuTorch toolchain — and to my teammates for the Android build, Mission Control UI, and the 3D rack work.

#OnDeviceAI #ExecuTorch #Qualcomm #Snapdragon #EdgeAI #ComputerVision #Hackathon

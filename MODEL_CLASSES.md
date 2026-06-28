# DC-Ops Model Classes Reference

Both **RetinaNet** (QNN HTP / NPU) and **YOLOv8n-seg** (XNNPACK / CPU) use the same
**16 DC-Ops classes**. RetinaNet's head has 17 outputs (16 classes + 1 background).

## Class List

| ID | Class | Notes |
|----|-------|-------|
| 0 | server rack | Full rack enclosure |
| 1 | compute tray | GPU/CPU trays (NVL72: 18 per rack) |
| 2 | NVLink switch tray | NVLink switches (NVL72: 9 per rack) |
| 3 | network switch | TOR / Ethernet / InfiniBand switches |
| 4 | power shelf | PSU banks |
| 5 | cable | Copper, fiber, power cables |
| 6 | network port | OSFP, QSFP, Ethernet, USB, HDMI, DP, VGA ports |
| 7 | LED indicator | Status LEDs on trays/switches |
| 8 | label | Serial numbers, asset tags |
| 9 | fan | Cooling fans |
| 10 | cooling manifold | Liquid cooling pipes/connections, heatsinks |
| 11 | cable cartridge | NVLink copper cable cartridge |
| 12 | power connector | Power plugs, bus bar connectors, PCIe power |
| 13 | drive bay | NVMe drive slots, HDD cages, disk drives |
| 14 | management port | BMC RJ45, serial console, IDSDM, microSD |
| 15 | DPU | BlueField DPU, ConnectX NIC, PCIe/SFP cards |

## Class Order (for code)

```python
DC_CLASSES = [
    "server rack",        # 0
    "compute tray",       # 1
    "NVLink switch tray", # 2
    "network switch",     # 3
    "power shelf",        # 4
    "cable",              # 5
    "network port",       # 6
    "LED indicator",      # 7
    "label",              # 8
    "fan",                # 9
    "cooling manifold",   # 10
    "cable cartridge",    # 11
    "power connector",    # 12
    "drive bay",          # 13
    "management port",    # 14
    "DPU",                # 15
]
```

```kotlin
val DC_CLASSES = arrayOf(
    "server rack", "compute tray", "NVLink switch tray", "network switch",
    "power shelf", "cable", "network port", "LED indicator",
    "label", "fan", "cooling manifold", "cable cartridge",
    "power connector", "drive bay", "management port", "DPU"
)
```

> **Note on indexing:** RetinaNet outputs are **1-indexed** (class 0 = background,
> class 1 = "server rack", ..., class 16 = "DPU"). Subtract 1 to map to the array above.
> YOLOv8n-seg outputs are **0-indexed** directly.

## Source Class Mapping

The 16 classes were mapped from 50+ source classes across 4 Roboflow datasets:

| Source classes | → DC-Ops class |
|----------------|----------------|
| USB, HDMI, RJ45, DP, SFP, XFP, VGA, ethernet, LAN, SATA/SAS, header ports | network port (6) |
| CPU socket, DIMM slot, CMOS battery, server (t200/t310) | compute tray (1) |
| NVMe slot, HDD cage, disk drive, drive bay | drive bay (13) |
| power supply, PSU | power shelf (4) |
| PCIe slot, NIC, NDC connector, riser, SFP port | DPU (15) |
| fan | fan (9) |
| heatsink | cooling manifold (10) |
| BMC, IDSDM, microSD, TPM connector | management port (14) |
| PCIe power, power connector | power connector (12) |

## Detection Reliability (IMPORTANT for demo)

Not all 16 classes have enough training data to detect reliably. Based on the
training data distribution and confusion matrix:

### ✅ Strong — point the camera at these
- **network port** (mAP50 0.91) — USB, HDMI, RJ45, etc.
- **compute tray** (0.92)
- **drive bay** (0.91)
- **power shelf** (0.83)
- **cooling manifold** (0.83)
- **DPU** (0.81)
- **LED indicator** (0.81)
- **fan** (0.52)

### ⚠️ Weak / rarely fires — few or zero training samples
- server rack, NVLink switch tray, network switch, cable,
  label, cable cartridge, power connector, management port

## Demo Tips

- Confidence threshold is set to **0.25** in the app (`ModelManager.kt`) so more
  detections show. Detections on real devices land around 30-50% confidence.
- Best demo props: anything with **visible ports** (mini PC, Raspberry Pi, USB hub,
  network switch) — the model reliably finds USB/HDMI/RJ45 as "network port".
- All port types currently map to one "network port" class — that's expected.

## Models

| Model | File | Backend | Size |
|-------|------|---------|------|
| RetinaNet-ResNet50-FPN | `dc_ops_retinanet_qnn.pte` | QNN HTP (NPU) | 36 MB |
| YOLOv8n-seg v3 | `dc_ops_yolov8n_seg.pte` | XNNPACK (CPU) | 13 MB |

Both on Hugging Face: https://huggingface.co/datasets/abhijitbetigeri/dc-ops-dataset

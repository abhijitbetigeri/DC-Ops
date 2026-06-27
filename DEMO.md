# DC-Ops — Demo Runbook (5 min)

Works **today** with `StubModelManager` (animated fake detections); swap in the
real `.pte` at M2 and the same script demos the real model.

## Setup (before you present)
1. Build & install: open `android-app/` in Android Studio → Run on the S25 Ultra.
2. Grant the camera permission on first launch.
3. Have a server rack (or a photo of one) ready to point at.

## Script
| Time | Beat | Action |
|------|------|--------|
| 0:00–1:00 | **Live detection** | Point at the rack. Polygons + labels appear, color-coded by class (green/amber/red LED, blue cable, magenta label). Detection chip shows the live count. |
| 1:00–2:00 | **Capture to audit log** | Tap **Capture** → toast "Logged N findings". Tap **Audit Log** → see timestamped, color-dotted entries. |
| 2:00–3:00 | **Confidence control** | Tap **Settings** → drag the confidence slider up → fewer, higher-quality detections surface. |
| 3:00–4:00 | **Offline proof** | Enable airplane mode → everything keeps working. Nothing ever left the device. |
| 4:00–5:00 | **Export + architecture** | In Audit Log tap **Export** → CSV via secure channel. Close with the ExecuTorch → QNN → NPU pipeline and latency numbers. |

## Talking points
- **On-device**: air-gapped, private, low-latency — no cloud round trip.
- **The seam**: model is swappable behind `ModelManager`; the app didn't change when we went from stub to the real `.pte`.
- **Audit trail**: every finding is logged to local SQLite and exportable.

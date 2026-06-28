# First-time build-box setup

Provisions a machine to compile DC-Ops QNN `.pte` models (ExecuTorch + Qualcomm QAIRT/QNN AOT
toolchain). After this, follow [`../MODEL_BUILD_GUIDE.md`](../MODEL_BUILD_GUIDE.md) to build a model.

## The hard constraint: the build host must be **Linux x86_64**

The QAIRT/QNN SDK ships only `x86_64-linux-clang` host libraries (Linux ELF binaries needing a
Linux kernel + glibc). So:

| Your machine | What to run | Notes |
|---|---|---|
| **Linux x86_64** | `bash setup/setup.sh` | Native. Fastest. |
| **Windows** | `./setup/setup.ps1` (PowerShell) | Bootstraps **WSL2 Ubuntu**, then runs `setup.sh` inside it. |
| **macOS** (Intel or Apple Silicon) | `bash setup/run-in-docker.sh` | macOS shell is Unix, **not** Linux — can't load the SDK's ELF libs. Runs in an x86_64 Docker container (Rosetta-emulated on Apple Silicon; slower but works). |
| Any of the above, app-only | *(skip setup)* | If you only build/deploy the **app**, you just need Android Studio + adb. Get `.pte` bundles from whoever has the Linux build box. |

> ❓ "Doesn't macOS have a Linux shell somewhere?" — Its Terminal is a **Unix** shell (zsh/bash),
> but Unix ≠ Linux. There's no Linux kernel, so it can't run the SDK's linux binaries. The Docker
> route gives you a real Linux x86_64 environment. (WSL is Windows-only — it doesn't exist on macOS.)

## One license-gated manual step: the QNN SDK

The QAIRT SDK can't be auto-downloaded (Qualcomm login / license). Get **QAIRT `2.46.0.260424`**
from <https://softwarecenter.qualcomm.com> (search "Qualcomm AI Engine Direct" / QAIRT). The
community zip downloads via a browser GET (a raw HEAD request returns 403). Then point setup at it:

- **Linux:**   `QNN_SDK_ZIP=/path/to/qairt-2.46.0.260424.zip bash setup/setup.sh`
- **Windows:** `./setup/setup.ps1 -QnnSdkZip "C:\Downloads\qairt-2.46.0.260424.zip"`
- **macOS:**   put the zip under the repo (e.g. `setup/qairt.zip`) then
  `QNN_SDK_ZIP=/work/setup/qairt.zip bash setup/run-in-docker.sh` (`/work` = the repo inside the container)

Everything else (miniforge/mamba, the `executorch` conda env with g++ 13 + cmake, NDK r26c,
ExecuTorch `main` + QNN backend build, the runtime libc++/libunwind fixes, and `~/et_setup/env.sh`)
is installed automatically. The script is **idempotent** — re-run it any time; it skips what's done.

## What you get

- `~/mambaforge` + conda env **`executorch`** (Python 3.10, g++ 13, cmake 3.31)
- `~/et_setup/qairt/2.46.0.260424` (QNN SDK), `~/et_setup/ndk/android-ndk-r26c`
- `~/executorch` built with the QNN backend (x86 host + aarch64-android)
- `~/et_setup/env.sh` — one-shot activator (`source ~/et_setup/env.sh`)
- project QNN scripts staged (LF) into `~/et_setup/`

The setup ends with a verification (`import executorch`, `QcomChipset.SM8750` resolves). If the
ExecuTorch QNN build hits a snag, the full error→fix postmortem is in
[`../EXECUTORCH_QNN_NPU_NOTES.md`](../EXECUTORCH_QNN_NPU_NOTES.md).

## Then build a model

```bash
source ~/et_setup/env.sh
MODEL_PT=<your.pt> OUT_PTE_BASE=<out> python scripts/qnn/export_yolo_backbone_qnn.py
# verify: ~3.8 MB and `strings <pte> | grep -c QnnBackend` >= 1   (13 MB / 0 = CPU build = wrong)
```

See [`../MODEL_BUILD_GUIDE.md`](../MODEL_BUILD_GUIDE.md) for the full recipe and deploy steps.

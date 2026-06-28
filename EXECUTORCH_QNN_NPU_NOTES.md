# Running YOLOv8-seg on the Snapdragon NPU with ExecuTorch + QNN — what broke and how we fixed it

Target: Galaxy S25 Ultra (Snapdragon 8 Elite, **SM8750 / Hexagon v79**).
Stack: ExecuTorch `main` (1.4.0a0) + QAIRT/QNN SDK **2.46.0.260424**, NDK r26c, host = WSL2 Ubuntu x86_64.
Model: Ultralytics **YOLOv8n-seg** (16 DC classes), `dc_ops_yolov8n_seg.pt`.

**TL;DR:** A straight QNN export of full YOLOv8-seg does **not** lower to HTP (the detection/seg
*head* uses ops the QNN backend can't quantize/compile). The fix is to **delegate only the
backbone** (export the raw pre-decode outputs) to the NPU and do the YOLO decode + NMS on the CPU
(Kotlin). The model then runs on the Hexagon NPU in **~4 ms**. The on-device *app* integration has
one remaining known wrinkle (skel loading from the app's private lib dir) — the model itself is
verified on the NPU via `qnn_executor_runner`.

---

## Error 1 — full YOLO-seg won't lower to QNN (3 cascading failures)

Exporting the whole model (det `[1,52,8400]` + proto) via `to_edge_transform_and_lower_to_qnn`:

1. **`AnnotateUnbind` pass crash** — `_passes/utils.py:get_quant_attrs`: `quant_node.target._schema`
   → `AttributeError: 'str' object has no attribute '_schema'`. The seg head's `unbind` is fed by a
   node whose `.target` is a `str`.
2. **`op_full` builder crash** — `builders/op_full.py`: `torch.full((list), Node, ...)` → `TypeError`.
   YOLO's head has an `aten.full` with a **dynamic (Node) fill value**; the builder assumes a scalar
   and crashes inside `is_node_supported` (aborting the whole partition).
3. **HTP graph-finalize failure** — `aten_slice : qti.aisw:StridedSlice failed 3110`,
   `mismatching datatypes 0x232 != 0x408/0x416`. Decoded `0x232 = FLOAT_32`, `0x408/0x416 =
   UFIXED_POINT_8/16` → a slice in the seg head keeps an **fp32 operand feeding a quantized op**
   (a quantization-annotation gap, **not** a bit-width issue — same failure under `use_8a8w` and `use_16a8w`).

### Fix: delegate the backbone only, decode the head on CPU
The failing ops live **only in producing the decoded `det[1,52,8400]`**. Export a wrapper that returns
just the **raw pre-decode outputs** and drop the decoded tensor — dead-code elimination then removes
the `unbind`/`full`/problematic-slice, and the pure-conv graph lowers to HTP cleanly:

```python
class RawWrap(nn.Module):          # forward(x) -> 3x(1,80,H,W) + (1,32,8400) + (1,32,160,160)
    def forward(self, x):
        o = self.m(x); ps, mc, pr = o[1][0], o[1][1], o[1][2]
        return ps[0], ps[1], ps[2], mc, pr
```

The YOLOv8-seg **decode** (DFL → distances, make_anchors, dist2bbox, class sigmoid, mask-coeff·proto,
NMS) is reimplemented in Kotlin (`ModelManager.decode()`), reading the 5 raw NPU outputs.
(Two small ExecuTorch QNN patches were also needed to get past the Python-level crashes:
a `str`-target guard in `_passes/annotate_unbind.py`, and a try/except in
`partition/qnn_partitioner.py::is_node_supported` so an unsupported op falls back to CPU instead of
aborting. These let the *backbone* lower; they aren't needed once the head is excluded, but are
harmless.)

Result: `dc_ops_yolov8n_seg_backbone_qnn.pte` (~3.8 MB, `QnnBackend`/HTP-delegated).

## Error 2 — on-device `QnnManager` SIGSEGV (version mismatch)

Loading the QNN `.pte` with the **prebuilt `1.0.0-qnn` AAR** crashed at
`QnnManager::QnnManager(...)` → `memmove` → SIGSEGV. Cause: the `.pte` was produced with ExecuTorch
**main**, but the runtime AAR was **1.0.0** — the `QnnContextCustomProtocol` differs between versions,
so the 1.0.0 runtime reads the main-format context binary with wrong offsets. (The portable/CPU
`.pte` survived this only because the portable format is version-stable; the QNN context binary is not.)

### Fix: build the AAR from the SAME ExecuTorch checkout that produced the `.pte`
`scripts/build_android_library.sh` with `EXECUTORCH_BUILD_QNN=ON` (needs `QNN_SDK_ROOT` +
`ANDROID_NDK`/`ANDROID_HOME`). Gotcha: it also needs a **full JDK** — system `java-17` was a JRE
(`Toolchain ... does not provide [JAVA_COMPILER]`); installed a real JDK via conda (`openjdk=17`) and
set `org.gradle.java.home`. The version-matched AAR fixed the `QnnManager` crash.

## Error 3 — `Failed to load skel error 4000` at execution

After init succeeded, `forward` failed: `QnnDsp Failed to load skel, error: 4000` /
`Failed to create device_handle ... 14001`. Two contributing causes:

- **`extractNativeLibs=false` (AGP default)** left `libQnnHtpV79Skel.so` *compressed inside the APK*,
  so it was never a real file the DSP could load. Fix: `packaging { jniLibs { useLegacyPackaging = true } }`.
- Even with the skel extracted to `/data/app/.../lib/arm64/`, the DSP's fastRPC still returns 4000 —
  the app-private lib dir is **not loadable by the cDSP** on a retail device. This is a **known
  ExecuTorch QNN-Android issue**:
  [#7492](https://github.com/pytorch/executorch/issues/7492),
  [#9084](https://github.com/pytorch/executorch/issues/9084),
  [#10993](https://github.com/pytorch/executorch/issues/10993).

### Root cause — CONFIRMED (not unsigned-PD, not the path)
We tried staging the skel to `/data/local/tmp/dcops_skel` (the DSP-readable path the runner uses) and
setting `ADSP_LIBRARY_PATH` to it from the app. **Still error 4000.** The on-device evidence pins it
down precisely:

1. **Unsigned PD is already what we request.** ExecuTorch's QNN backend defaults to unsigned PD
   (`serialization/qc_schema.py:191` → `pd_session = kHtpUnsignedPd`; `HtpDevicePlatformInfoConfig.cpp`
   comment "default value is unsigned pd"). So "enable unsigned PD" is **not** a missing knob — the
   `.pte` already asks for it. The runner and the app use the same AAR, so they request the *same* PD
   session type. The differentiator is **permission to establish** the session, not the request.
2. **The DSP firmware supports unsigned PD on this exact phone.** logcat shows the Samsung camera HAL
   (a system/vendor process) doing it successfully: `remote_session_control Unsigned PD enable 1
   request for domain 7` → `Created user PD on domain 7 ... Unsigned:Y`. So it's not a firmware-wide
   block.
3. **The wall is the `untrusted_app` SELinux context (uid 10380) reaching the cDSP.** `error 4000 =
   loadRemoteSymbols failed` is a **DSP-side** rejection (the skel was found and shipped; the DSP
   refused to load it into the PD), *not* file-not-found. The runner runs as the `shell` domain, which
   is allowed the cDSP session; our app runs as `untrusted_app`, which is not. Path staging can't fix
   this — and worse, `/data/local/tmp` is the `shell_data_file` SELinux type, which `untrusted_app` is
   itself denied from reading, so staging the skel there for the *app* was a dead end from the start.

**Bottom line:** on a non-rooted retail S25, a normal third-party app (`untrusted_app`) cannot open an
unsigned-PD cDSP session, so the QNN/HTP `.pte` cannot run from app context — regardless of where the
skel lives or what `ADSP_LIBRARY_PATH` is set to. The model itself is correct and runs on the NPU; the
limitation is Android/Qualcomm device policy, not our build. Getting NPU-in-app needs one of: an
OEM/Qualcomm-signed PD, a `shell`/system-context helper process the app talks to (the runner pattern
wrapped as a service), or a device where the untrusted_app→cDSP policy is relaxed. This is the right,
specific question for the ExecuTorch team (reference the issues above and these three observations).

### Verified workaround: run from `/data/local/tmp` (DSP-accessible)
Pushing the `.pte` + `qnn_executor_runner` + QNN libs (incl. the v79 skel) to `/data/local/tmp` and
running with `LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=.` **works** — the skel loads and the model runs on
the NPU:

```
qnn_executor_runner: Qnn backend type 2 (HTP)
1 inference took 4.153 ms        # on Hexagon v79
outputs: (1,80,80,80)(1,80,40,40)(1,80,20,20)(1,32,8400)(1,32,160,160)
decode -> "cable 0.46"           # real DC-class detection
```

Open item for the **app** (vs the runner): getting the cDSP to load the skel from the app context.
**Confirmed** to be the `untrusted_app`→cDSP unsigned-PD policy wall, not the path or the request type
— see "Root cause — CONFIRMED" above. Tried & ruled out: skel in jniLibs (`useLegacyPackaging`),
skel staged to `/data/local/tmp/dcops_skel`, `ADSP_LIBRARY_PATH` set from `ModelManager.init`. All
produce the identical error 4000.

## SOLUTION — NPU in the app via a shell-context bridge (WORKING, verified IRL)
Since `untrusted_app` can't open the cDSP but `shell` can, we move *only the inference* into a
shell-launched process and let the app talk to it over loopback TCP. Real NPU polygons now render in
the `com.dcops.ar.qnn` app on the unrooted retail S25 (verified: live camera ~4 fps, and test image →
`cable 53%` polygon overlay).

```
 com.dcops.ar.qnn (untrusted_app)            yolo_npu_server (shell uid 2000, /data/local/tmp)
 ───────────────────────────────            ────────────────────────────────────────────────
 camera frame → 640x640x3 f32 /255  ──TCP──►  recv 4.9MB → write input.raw
 (127.0.0.1:8765)                             → exec qnn_executor_runner on Hexagon v79 (~4ms)
 decode (DFL/anchors/NMS/mask) ◄──TCP────────  read 5 out tensors → send 7.04MB back
 → draw polygons
```

Why it works: the server runs as `shell` (uid 2000), the one context the retail cDSP grants an
unsigned-PD session — exactly like the verified runner. The app is a normal app; it never touches the
DSP. Loopback TCP is the only clean cross-uid channel (the app can't read `/data/local/tmp`, and
`shell` can't read the app's private dir on a user build).

Pieces (all in `~/et_setup/`, repo copies in `scripts/qnn/`):
- **`yolo_npu_server.c`** — dependency-free NDK-compiled TCP relay. Fixed framing: client sends
  `1x3x640x640` f32 (4,915,200 B); server replies 5 tensors concatenated (7,040,000 B):
  `(1,80,80,80)(1,80,40,40)(1,80,20,20)(1,32,8400)(1,32,160,160)`. Per frame it writes `input.raw`,
  runs `qnn_executor_runner --model_path model.pte --input_list_path input_list.txt
  --output_folder_path out`, and ships back `out/output_0_{0..4}.raw`.
- **`build_deploy_npu_server.sh`** — compiles the server (NDK r26c `aarch64-linux-android26-clang`),
  stages server + runner + `libqnn_executorch_backend.so` + QNN runtime libs + v79 skel + `.pte`
  into `/data/local/tmp/dcops_yolo_qnn`, and launches the daemon.
- **`start_npu_server.sh`** — (re)launch the daemon. **Run once per device reboot** (the shell uid is
  inherited from adb, so the daemon is `setsid`-detached and survives adb disconnect, but not a reboot).
- App side: `android-app-qnn/.../ModelManager.kt` now opens a persistent socket to `127.0.0.1:8765`
  (needs `<uses-permission android:name="android.permission.INTERNET"/>`), sends the frame, reads the
  5 outputs, and runs the same Kotlin decode. No `Module.load` of the QNN `.pte` in-app anymore.

### Real-time: persistent server (load-once) — DONE
The original relay spawned `qnn_executor_runner` per frame: ~190 ms/frame (~5 fps), dominated by
re-creating the QNN context + reloading the 87 MB `libQnnHtpPrepare.so` every call. Replaced by
**`qnn_socket_server.cpp`** — a C++ server that links the ExecuTorch **Module API**
(`extension/module`) + QNN backend, loads the `.pte` **once**, and runs `forward()` per socket frame.
Built as a target in `examples/qualcomm/executor_runner/CMakeLists.txt` (`build_socket_server.sh`),
deployed by `push_persistent.sh` + `start_npu_server.sh`.

Profiling (instrumented both sides — server logs `conv/forward/copy/send`, app logs `prep/net/decode`):

| stage | per-frame |
|-------|-----------|
| server forward (NPU) | **~3 ms** (was ~190 ms via spawn) |
| server conv (RGBA→NCHW float) + copy + send | ~0.2 + 0.4 + 1.5 ms → server total **~5 ms** |
| app prep | **0.1 ms** (was 31 ms — see below) |
| app net (8.7 MB round-trip over loopback) | ~6.5 ms |
| app decode (Kotlin DFL/anchors/NMS/mask) | ~4 ms |
| **app end-to-end** | **~11–23 ms (≈ 43–90 fps)** |

Two optimizations got there:
1. **Persistent model** (load-once Module) → forward 190 ms → 3 ms.
2. **RGBA wire input.** The app's preprocessing was 31 ms (a 1.2 M-element Kotlin loop doing
   `getPixels` + per-channel `/255` into a FloatBuffer). Replaced with `Bitmap.copyPixelsToBuffer`
   (one native memcpy → **0.1 ms**); the server now receives raw RGBA8888 (640×640×4 = 1.6 MB, also
   4× smaller than float NCHW) and does the RGBA→NCHW-float/255 conversion in C++ (~0.2 ms).

Result is well past the ~30 fps camera rate, so it's real-time.

### Server-side decode + model-swap — DONE (≈ 8 ms / 125 fps)
The next lever (now taken): the full YOLOv8-seg decode (DFL → dist2bbox, sigmoid, area filter,
per-class NMS, mask→convex-hull) was **ported into the C++ server**, which now returns only the final
detections (~272 B/frame vs 7 MB of raw tensors). App `parseDetections` reads
`int32 payload_len | int32 num_dets | per-det {int32 class_id, float score, int32 n_pts, n_pts×(float x,float y)}`.
App end-to-end dropped to **~8 ms (≈ 125 fps)**; the 1.6 MB RGBA input upload now dominates `net`.

The server is also **model-driven** — nothing about the model is hardcoded anymore:
- At load it reads optional `meta.txt` (`type=`, `reg_max=`, `input_size=`) + `labels.txt` (one per
  line) from the model's directory (falls back to the historical 16-class/640 defaults), then
  **derives** `input_w/h`, `num_classes (= ps_ch − 4·reg_max)`, `num_mask`, `mask_res`, the per-head
  `(grid, stride)` scales and the anchor count **from the `.pte`'s tensor shapes**. Startup logs e.g.
  `model: type=yolov8_seg input=640x640 classes=16 masks=32 mask_res=160 scales=80/8,40/16,20/32 anchors=8400`.
- Decode is dispatched by `type` (`decode_yolov8_seg` implemented). A genuinely different family =
  add one `decode_<type>()` + a switch case — the only C++ edit a new architecture needs.
- **Handshake**: on connect the server sends `int32 magic=0x4D4F444C ("MODL"), input_w, input_h,
  num_classes, then num_classes×(int32 len, UTF-8 label)`. The app reads it first and configures its
  input size + label list dynamically (`ModelManager.readHandshakeLocked`/`applySpec`), falling back
  to compiled defaults against an old, handshake-less server.

**Swapping a model** (no rebuild for any YOLOv8-seg variant — different classes or input size):
- A model is a **bundle** = `model.pte` + `model.json` (`{type, input_size, reg_max, labels[]}`). The
  committed default is `models/dcops-v1/`.
- `scripts/qnn/swap_model.sh <bundle-dir | model.pte | hf-url> [model.json]` generates `meta.txt` +
  `labels.txt` from the json, `adb push`es `model.pte`/`meta.txt`/`labels.txt` to
  `/data/local/tmp/dcops_yolo_qnn/`, restarts the server, and verifies. `scripts/qnn/deploy_default.sh`
  runs it on the repo default for one-touch fresh-device deploy. (Run the LF copies in `~/et_setup/`;
  the repo copies are CRLF.)
- Note: `*.pte` is gitignored, so `models/dcops-v1/model.pte` is present on disk but **not committed** —
  re-add a `!models/dcops-v1/model.pte` negation or use Git LFS if you want clone-and-deploy.

**Caveat / honesty:** this is a *tethered/dev-mode* NPU path — the helper must be started from adb
once per boot. A truly standalone consumer app on the NPU still needs OEM/Qualcomm signing or root.
But for the demo it's the real thing: live camera → Hexagon NPU → polygons in the app UI.

To swap in a better-calibrated model: `adb push new.pte /data/local/tmp/dcops_yolo_qnn/model.pte`
then re-run `start_npu_server.sh`. No app rebuild needed.

## Quantization note
INT8 PTQ with only 8 calibration images cut recall (1 detection vs several from the FP CPU model).
For production: calibrate on a representative DC image set, consider per-channel weights / `use_16a8w`
(w8a16, YOLO's recommended HTP precision), and validate mAP after quantization.

## Reproduce (host = WSL, env via `~/et_setup/env.sh`)
1. Export backbone: `~/et_setup/export_yolo_backbone_qnn.py` → `dc_ops_yolov8n_seg_backbone_qnn.pte`
2. Build matching AAR: `~/et_setup/fix_jdk_build_aar.sh` → `~/dcops_qnn/aar/executorch.aar`
3. Verify on NPU: `~/et_setup/run_yolo_qnn_runner.sh` (push to /data/local/tmp + run) then
   `~/et_setup/decode_qnn_outputs.py`
4. App: `android-app-qnn/` (QNN AAR in `app/libs/`, QNN libs in `jniLibs/arm64-v8a/`, `.pte` in
   `assets/`, `useLegacyPackaging=true`, `ADSP_LIBRARY_PATH` set in `ModelManager.init`).

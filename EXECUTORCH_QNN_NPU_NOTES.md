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
  the app-private lib dir is **not loadable by the cDSP** on a retail device (SELinux + unsigned-PD).
  This is a **known ExecuTorch QNN-Android issue**:
  [#7492](https://github.com/pytorch/executorch/issues/7492),
  [#9084](https://github.com/pytorch/executorch/issues/9084),
  [#10993](https://github.com/pytorch/executorch/issues/10993).

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
Likely needs unsigned-PD enablement and/or staging the skel to a DSP-accessible path. Good question
for the ExecuTorch team (reference the issues above).

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

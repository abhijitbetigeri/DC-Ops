# Demo backup

`dc-ops-qnn-demo.apk` — prebuilt debug APK of the QNN/NPU app (`com.dcops.ar.qnn`),
the working version with the **live-camera fix** (RGBA frame conversion + upright rotation)
and **per-object detection stabilizer** (IoU tracking + TTL hold + confidence hysteresis).

Use this as a fallback if a rebuild isn't possible during the demo:

```
adb install -r demo/dc-ops-qnn-demo.apk
```

Requires the on-device NPU server running (`scripts/qnn/start_npu_server.sh`, once per boot)
and a deployed model bundle (`scripts/qnn/deploy_default.sh`). The app connects to it over
127.0.0.1:8765.

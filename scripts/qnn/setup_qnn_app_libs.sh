#!/bin/bash
set -e
APP="/mnt/c/Users/Rashi/AndroidStudioProjects/DC-Ops/android-app-qnn/app"
QNN="$HOME/et_setup/qairt/2.46.0.260424"
JNI="$APP/src/main/jniLibs/arm64-v8a"
ASSETS="$APP/src/main/assets"
mkdir -p "$JNI" "$ASSETS"

# Qualcomm QNN runtime libs the ExecuTorch QNN backend dlopens on-device (HTP / v79)
cp "$QNN/lib/aarch64-android/libQnnHtp.so"            "$JNI/"
cp "$QNN/lib/aarch64-android/libQnnSystem.so"         "$JNI/"
cp "$QNN/lib/aarch64-android/libQnnHtpV79Stub.so"     "$JNI/"
cp "$QNN/lib/aarch64-android/libQnnHtpPrepare.so"     "$JNI/"
cp "$QNN/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" "$JNI/"

# the QNN/NPU model (raw-output backbone)
cp "$HOME/executorch/dcops_qnn/dc_ops_yolov8n_seg_backbone_qnn.pte" "$ASSETS/"
# drop the old CPU model from this QNN variant (keep test_rack.jpg for the diagnostic)
rm -f "$ASSETS/dc_ops_yolov8n_seg.pte" 2>/dev/null || true

echo "=== jniLibs ==="; ls -lh "$JNI"
echo "=== assets ==="; ls -lh "$ASSETS"

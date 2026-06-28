#!/bin/bash
# Build the ExecuTorch Android AAR (with QNN backend) from OUR ~/executorch checkout,
# so its QNN runtime matches the .pte we exported (fixes the QnnManager version-mismatch SIGSEGV).
SDK="$HOME/android-sdk"

# --- 1. Linux Android SDK from the already-downloaded cmdline tools ---
if [ ! -x "$SDK/cmdline-tools/latest/bin/sdkmanager" ]; then
  mkdir -p "$SDK/cmdline-tools"
  ( cd "$SDK/cmdline-tools" && rm -rf latest cmdline-tools && \
    unzip -q "$HOME/commandlinetools-linux-11076708_latest.zip" && mv cmdline-tools latest )
fi
export ANDROID_HOME="$SDK" ANDROID_SDK="$SDK"
yes | "$SDK/cmdline-tools/latest/bin/sdkmanager" --sdk_root="$SDK" --licenses >/dev/null 2>&1 || true
"$SDK/cmdline-tools/latest/bin/sdkmanager" --sdk_root="$SDK" \
    "platform-tools" "platforms;android-35" "build-tools;35.0.0" 2>&1 | tail -3
echo "java: $(java -version 2>&1 | head -1)"

# --- 2. build env (conda g++13/cmake, QNN SDK, NDK) ---
source "$HOME/et_setup/env.sh" >/dev/null 2>&1
export ANDROID_NDK="$ANDROID_NDK_ROOT"
export ANDROID_ABIS="arm64-v8a"                 # device ABI only (faster)
export EXECUTORCH_BUILD_EXTENSION_LLM=OFF        # not needed for YOLO; trims the build
export BUILD_AAR_DIR="$HOME/dcops_qnn/aar"; mkdir -p "$BUILD_AAR_DIR"

cd "$EXECUTORCH_ROOT" || exit 2
echo "AAR_BUILD_START $(date)  QNN_SDK_ROOT=$QNN_SDK_ROOT  NDK=$ANDROID_NDK"
bash scripts/build_android_library.sh || echo "build_android_library.sh rc=$?"

echo "=== locate AAR ==="
ls -lh "$BUILD_AAR_DIR/executorch.aar" 2>/dev/null
find "$EXECUTORCH_ROOT/extension/android" -name '*.aar' -exec ls -lh {} \; 2>/dev/null
echo "AAR_BUILD_DONE $(date)"

#!/bin/bash
# The gradle AAR step failed: system java-17 is a JRE (no javac). Install a full JDK (conda, no sudo),
# point gradle at it, and finish the AAR (reusing the already-built native libs if present).
~/mambaforge/bin/mamba install -y -n executorch -c conda-forge openjdk=17 2>&1 | tail -3
source "$HOME/et_setup/env.sh" >/dev/null 2>&1

JDK="$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")"
echo "JDK_HOME=$JDK"
"$JDK/bin/javac" -version || { echo "no javac"; exit 1; }
export JAVA_HOME="$JDK"; export PATH="$JDK/bin:$PATH"
mkdir -p "$HOME/.gradle"; echo "org.gradle.java.home=$JDK" > "$HOME/.gradle/gradle.properties"

export ANDROID_HOME="$HOME/android-sdk" ANDROID_SDK="$HOME/android-sdk"
export ANDROID_NDK="$ANDROID_NDK_ROOT" ANDROID_ABIS="arm64-v8a" EXECUTORCH_BUILD_EXTENSION_LLM=OFF
export BUILD_AAR_DIR="$HOME/dcops_qnn/aar"; mkdir -p "$BUILD_AAR_DIR"
cd "$EXECUTORCH_ROOT" || exit 2

# Try gradle-only first (native libs already staged from the prior run)
if [ -d cmake-out-android-so ] && [ -n "$(find cmake-out-android-so -name '*.so' 2>/dev/null | head -1)" ]; then
  echo "=== native libs staged -> gradle assembleDebug only ==="
  ( cd extension/android && ANDROID_HOME="$HOME/android-sdk" ./gradlew :executorch_android:assembleDebug --no-daemon ) \
    && cp extension/android/executorch_android/build/outputs/aar/executorch_android-debug.aar "$BUILD_AAR_DIR/executorch.aar"
fi
if [ ! -f "$BUILD_AAR_DIR/executorch.aar" ]; then
  echo "=== full build_android_library.sh (native + aar) ==="
  bash scripts/build_android_library.sh
fi
echo "=== RESULT ==="
ls -lh "$BUILD_AAR_DIR/executorch.aar" 2>/dev/null || find extension/android -name '*.aar' -exec ls -lh {} \; 2>/dev/null

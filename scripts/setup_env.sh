#!/bin/bash
# DC-Ops environment setup script
# Run: source scripts/setup_env.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Python virtual environment
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "Creating virtual environment with Python 3.12..."
    python3.12 -m venv "$PROJECT_ROOT/.venv"
fi
source "$PROJECT_ROOT/.venv/bin/activate"

# Java
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export PATH="$JAVA_HOME/bin:$PATH"

# Android
export ANDROID_HOME=/opt/homebrew/share/android-commandlinetools
export ANDROID_NDK_ROOT="$ANDROID_HOME/ndk/26.3.11579264"
export PATH="$ANDROID_HOME/platform-tools:$PATH"

# Qualcomm QAIRT SDK — update this path after downloading
if [ -z "$QNN_SDK_ROOT" ]; then
    echo "WARNING: QNN_SDK_ROOT not set. Set it to your QAIRT SDK path, e.g.:"
    echo "  export QNN_SDK_ROOT=~/qualcomm/qairt/2.46.0.260424"
fi

echo "DC-Ops environment ready."
echo "  Python:      $(python --version)"
echo "  PyTorch:     $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "  Java:        $(java -version 2>&1 | head -1)"
echo "  NDK:         $ANDROID_NDK_ROOT"
echo "  QNN SDK:     ${QNN_SDK_ROOT:-NOT SET}"

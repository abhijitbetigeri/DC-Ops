#!/bin/bash
# ExecuTorch + Qualcomm QNN (QAIRT) 2.46 build environment.
# Usage:  source ~/et_setup/env.sh
# Targets: Galaxy S25 Ultra (SM8750 / Hexagon v79), AOT build on WSL2 host.

# --- conda env ---
source "$HOME/mambaforge/etc/profile.d/conda.sh"
conda activate executorch

# --- roots ---
export EXECUTORCH_ROOT="$HOME/executorch"
export QNN_SDK_ROOT="$HOME/et_setup/qairt/2.46.0.260424"
export ANDROID_NDK_ROOT="$HOME/et_setup/ndk/android-ndk-r26c"

# --- python / linker paths ---
export PYTHONPATH="$EXECUTORCH_ROOT/..:$PYTHONPATH"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# --- C/C++ compiler: prefer conda gcc/g++ 13 (installed via conda-forge, no sudo) ---
# ExecuTorch's Qualcomm AOT build requires g++ >= 13; Ubuntu 22.04 ships 11.
_CONDA_GXX="$(ls "$CONDA_PREFIX"/bin/*-linux-gnu-g++ 2>/dev/null | head -1)"
_CONDA_GCC="$(ls "$CONDA_PREFIX"/bin/*-linux-gnu-gcc 2>/dev/null | head -1)"
if [ -n "$_CONDA_GXX" ] && [ -n "$_CONDA_GCC" ]; then
  export CXX="$_CONDA_GXX"
  export CC="$_CONDA_GCC"
  # expose plain g++/gcc names (some sub-builds call them directly)
  mkdir -p "$HOME/et_setup/compiler-bin"
  ln -sf "$_CONDA_GXX" "$HOME/et_setup/compiler-bin/g++"
  ln -sf "$_CONDA_GXX" "$HOME/et_setup/compiler-bin/c++"
  ln -sf "$_CONDA_GCC" "$HOME/et_setup/compiler-bin/gcc"
  ln -sf "$_CONDA_GCC" "$HOME/et_setup/compiler-bin/cc"
  export PATH="$HOME/et_setup/compiler-bin:$PATH"
fi
unset _CONDA_GXX _CONDA_GCC

# --- QNN's own env setup (sets QNN_SDK_ROOT-relative vars) ---
if [ -f "$QNN_SDK_ROOT/bin/envsetup.sh" ]; then
  # re-export QNN_SDK_ROOT afterwards in case envsetup overrides it
  _ETX_QNN_ROOT="$QNN_SDK_ROOT"
  source "$QNN_SDK_ROOT/bin/envsetup.sh" >/dev/null 2>&1
  export QNN_SDK_ROOT="$_ETX_QNN_ROOT"
  unset _ETX_QNN_ROOT
fi

echo "[env] executorch env active"
echo "[env] QNN_SDK_ROOT     = $QNN_SDK_ROOT"
echo "[env] ANDROID_NDK_ROOT = $ANDROID_NDK_ROOT"
echo "[env] g++             = $(g++ --version 2>/dev/null | head -1)"

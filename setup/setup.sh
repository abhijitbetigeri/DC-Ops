#!/usr/bin/env bash
# =============================================================================
# DC-Ops build-box setup — ExecuTorch + Qualcomm QNN (QAIRT) AOT toolchain.
#
# This provisions a FRESH machine to the point where you can run
#   scripts/qnn/export_yolo_backbone_qnn.py
# and produce a QNN .pte that runs on the Galaxy S25 Ultra NPU.
#
# Supported host: **Linux x86_64** (the QNN SDK ships x86_64-linux-clang libs).
#   - On Windows: run setup.ps1 instead — it bootstraps WSL2 Ubuntu and calls this.
#   - On macOS:   the QNN AOT build CANNOT run locally (no macOS/arm64 QNN host).
#                 This script will detect Darwin and explain the only options.
#
# Idempotent: re-running skips anything already in place. Safe to re-run.
#
# Verified steps come from EXECUTORCH_QNN_NPU_NOTES.md + the project's build notes.
# The ExecuTorch QNN build is long (~20-40 min) and the QNN SDK download is
# license-gated (manual) — see "QNN SDK" below.
# =============================================================================
set -euo pipefail

# ---- version pins (override via env if you must) ----------------------------
QNN_VER="${QNN_VER:-2.46.0.260424}"
NDK_VER="${NDK_VER:-r26c}"
PY_VER="${PY_VER:-3.10}"
ET_REF="${ET_REF:-main}"
ENV_NAME="${ENV_NAME:-executorch}"

# ---- layout (matches scripts/qnn/env.sh) ------------------------------------
ET_SETUP="$HOME/et_setup"
ET_ROOT="$HOME/executorch"
CONDA_HOME="$HOME/mambaforge"
QNN_SDK_ROOT="$ET_SETUP/qairt/$QNN_VER"
NDK_ROOT="$ET_SETUP/ndk/android-ndk-$NDK_VER"

# Repo root = parent of this script's dir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

c()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }    # phase header
ok() { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }
warn(){ printf '\033[1;33m  ! %s\033[0m\n' "$*"; }
die(){ printf '\033[1;31m  ✗ %s\033[0m\n' "$*" >&2; exit 1; }

# =============================================================================
# 0. Host check
# =============================================================================
c "Host check"
OS="$(uname -s)"; ARCH="$(uname -m)"
echo "  OS=$OS ARCH=$ARCH"

if [ "$OS" = "Darwin" ]; then
  cat <<'EOF'

  macOS's Terminal is a Unix shell, but it is NOT Linux — and the QAIRT SDK ships
  only x86_64-linux-clang host libraries (ELF binaries that need a Linux kernel +
  glibc). macOS's Darwin userland can't load them, so the native shell can't run
  the QNN compiler. You need a real Linux x86_64 environment. Easiest:

    >>> RUN IT IN DOCKER (works on Intel + Apple Silicon):
          bash setup/run-in-docker.sh
        This launches an x86_64 Ubuntu container (Rosetta-emulated on Apple
        Silicon — slower but fine; the compile is CPU-bound) and runs this
        script inside it, with the repo mounted and the heavy build cached in a
        named volume. The .pte lands back on your Mac under models/.

  Other options:
    - A Linux x86_64 box (bare metal / VM / cloud) — run THIS script there.
    - App-only work (no new models)? Android Studio + adb on macOS is enough;
      get .pte bundles from whoever has the Linux build box.

  Stopping — there is nothing to install in the macOS shell itself.
EOF
  exit 0
fi

[ "$OS" = "Linux" ]   || die "Unsupported OS '$OS' (need Linux x86_64; on Windows use setup.ps1)."
[ "$ARCH" = "x86_64" ] || die "Unsupported arch '$ARCH' (QNN host libs are x86_64 only)."
if grep -qi microsoft /proc/version 2>/dev/null; then ok "Running under WSL2"; else ok "Native Linux"; fi
mkdir -p "$ET_SETUP"

# =============================================================================
# 1. System deps that genuinely need apt (git, build basics, unzip, curl)
#    g++/cmake come from conda (no sudo) — see step 3.
# =============================================================================
c "Base tools (git, curl, unzip)"
need_apt=()
for t in git curl unzip; do command -v "$t" >/dev/null 2>&1 || need_apt+=("$t"); done
if [ "${#need_apt[@]}" -gt 0 ]; then
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -y && sudo apt-get install -y "${need_apt[@]}"
  else
    die "Missing tools (${need_apt[*]}) and no sudo. Install them and re-run."
  fi
fi
ok "git/curl/unzip present"

# =============================================================================
# 2. Miniforge/mamba  ->  ~/mambaforge
# =============================================================================
c "Conda (miniforge/mamba)"
if [ ! -x "$CONDA_HOME/bin/conda" ]; then
  echo "  installing miniforge to $CONDA_HOME ..."
  curl -fsSL -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$CONDA_HOME"
fi
# shellcheck disable=SC1091
source "$CONDA_HOME/etc/profile.d/conda.sh"
ok "conda at $CONDA_HOME"

# =============================================================================
# 3. The 'executorch' env: python + g++13 + cmake + the QNN runtime libs.
#    (Ubuntu's apt g++/cmake are too old; conda-forge avoids sudo entirely.)
# =============================================================================
c "Conda env '$ENV_NAME'"
if ! conda env list | grep -qE "^\s*$ENV_NAME\s"; then
  mamba create -y -n "$ENV_NAME" "python=$PY_VER"
fi
# Toolchain + the libs QNN's x86 host libs need at runtime (libc++/libc++abi/libunwind).
mamba install -y -n "$ENV_NAME" -c conda-forge \
  "gxx_linux-64=13" "gcc_linux-64=13" "cmake=3.31" make ninja \
  libcxx libcxxabi libunwind
conda activate "$ENV_NAME"
# QNN wants libunwind.so.1 (LLVM soname); conda-forge ships libunwind.so.8 (nongnu) -> symlink.
if [ -e "$CONDA_PREFIX/lib/libunwind.so.8" ] && [ ! -e "$CONDA_PREFIX/lib/libunwind.so.1" ]; then
  ln -sf "$CONDA_PREFIX/lib/libunwind.so.8" "$CONDA_PREFIX/lib/libunwind.so.1"
  ok "symlinked libunwind.so.1 -> libunwind.so.8"
fi
ok "env '$ENV_NAME' ready (python $PY_VER, g++ 13, cmake 3.31)"

# =============================================================================
# 4. Android NDK r26c  ->  ~/et_setup/ndk/android-ndk-r26c
# =============================================================================
c "Android NDK $NDK_VER"
if [ ! -d "$NDK_ROOT" ]; then
  mkdir -p "$ET_SETUP/ndk"
  echo "  downloading NDK $NDK_VER ..."
  curl -fSL -o /tmp/ndk.zip "https://dl.google.com/android/repository/android-ndk-$NDK_VER-linux.zip"
  unzip -q /tmp/ndk.zip -d "$ET_SETUP/ndk"
fi
[ -d "$NDK_ROOT" ] && ok "NDK at $NDK_ROOT" || die "NDK extract failed ($NDK_ROOT not found)"

# =============================================================================
# 5. QNN / QAIRT SDK  ->  ~/et_setup/qairt/<ver>   (LICENSE-GATED, manual)
# =============================================================================
c "QNN/QAIRT SDK $QNN_VER"
if [ ! -d "$QNN_SDK_ROOT" ]; then
  if [ -n "${QNN_SDK_ZIP:-}" ] && [ -f "$QNN_SDK_ZIP" ]; then
    echo "  extracting $QNN_SDK_ZIP ..."
    mkdir -p "$ET_SETUP/qairt"
    unzip -q "$QNN_SDK_ZIP" -d "$ET_SETUP/qairt"
    # The zip usually extracts to a versioned dir; normalize if needed.
    [ -d "$QNN_SDK_ROOT" ] || warn "Extracted, but $QNN_SDK_ROOT not found — check the dir name under $ET_SETUP/qairt and rename to $QNN_VER."
  else
    cat <<EOF

  The QAIRT/QNN SDK is license-gated and cannot be auto-downloaded.
  1. Get QAIRT $QNN_VER from Qualcomm:
       https://softwarecenter.qualcomm.com  (search "Qualcomm AI Engine Direct" / QAIRT)
     (Community zip downloads via a GET request; a HEAD request returns 403 — use a browser.)
  2. Then re-run with the zip path:
       QNN_SDK_ZIP=/path/to/qairt-$QNN_VER.zip bash setup/setup.sh
     ...or extract it yourself so this exists:
       $QNN_SDK_ROOT/bin/envsetup.sh

  Stopping here — the rest of the build needs the SDK.
EOF
    exit 1
  fi
fi
[ -f "$QNN_SDK_ROOT/bin/envsetup.sh" ] && ok "QNN SDK at $QNN_SDK_ROOT" \
  || die "QNN SDK incomplete: $QNN_SDK_ROOT/bin/envsetup.sh missing"

# =============================================================================
# 6. Write ~/et_setup/env.sh (the one-shot activator everything else sources)
# =============================================================================
c "Generating $ET_SETUP/env.sh"
cat > "$ET_SETUP/env.sh" <<EOF
#!/bin/bash
# AUTO-GENERATED by setup/setup.sh — ExecuTorch + QNN (QAIRT $QNN_VER) build env.
# Usage:  source ~/et_setup/env.sh
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate $ENV_NAME

export EXECUTORCH_ROOT="$ET_ROOT"
export QNN_SDK_ROOT="$QNN_SDK_ROOT"
export ANDROID_NDK_ROOT="$NDK_ROOT"

export PYTHONPATH="\$EXECUTORCH_ROOT/..:\$PYTHONPATH"
export LD_LIBRARY_PATH="\$QNN_SDK_ROOT/lib/x86_64-linux-clang:\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"

# Prefer conda's gcc/g++ 13 (ExecuTorch's Qualcomm AOT build needs g++ >= 13).
_GXX="\$(ls "\$CONDA_PREFIX"/bin/*-linux-gnu-g++ 2>/dev/null | head -1)"
_GCC="\$(ls "\$CONDA_PREFIX"/bin/*-linux-gnu-gcc 2>/dev/null | head -1)"
if [ -n "\$_GXX" ] && [ -n "\$_GCC" ]; then
  export CXX="\$_GXX"; export CC="\$_GCC"
  mkdir -p "$ET_SETUP/compiler-bin"
  ln -sf "\$_GXX" "$ET_SETUP/compiler-bin/g++"; ln -sf "\$_GXX" "$ET_SETUP/compiler-bin/c++"
  ln -sf "\$_GCC" "$ET_SETUP/compiler-bin/gcc"; ln -sf "\$_GCC" "$ET_SETUP/compiler-bin/cc"
  export PATH="$ET_SETUP/compiler-bin:\$PATH"
fi
unset _GXX _GCC

if [ -f "\$QNN_SDK_ROOT/bin/envsetup.sh" ]; then
  _Q="\$QNN_SDK_ROOT"; source "\$QNN_SDK_ROOT/bin/envsetup.sh" >/dev/null 2>&1; export QNN_SDK_ROOT="\$_Q"; unset _Q
fi
echo "[env] executorch env active | QNN=\$QNN_SDK_ROOT | NDK=\$ANDROID_NDK_ROOT | g++ \$(g++ --version 2>/dev/null | head -1)"
EOF
chmod +x "$ET_SETUP/env.sh"
# Keep the repo copy in sync so scripts/qnn/env.sh references work too.
cp "$ET_SETUP/env.sh" "$REPO_ROOT/scripts/qnn/env.sh" 2>/dev/null || true
ok "wrote env.sh"

# shellcheck disable=SC1091
source "$ET_SETUP/env.sh"

# =============================================================================
# 7. ExecuTorch (clone main) + install + QNN backend build
# =============================================================================
c "ExecuTorch (clone + build, ~20-40 min)"
if [ ! -d "$ET_ROOT/.git" ]; then
  git clone --recursive https://github.com/pytorch/executorch.git "$ET_ROOT"
  ( cd "$ET_ROOT" && git checkout "$ET_REF" && git submodule sync && git submodule update --init --recursive )
fi
cd "$ET_ROOT"
echo "  ./install_executorch.sh ..."
./install_executorch.sh
echo "  ./backends/qualcomm/scripts/build.sh (x86 host + aarch64-android) ..."
./backends/qualcomm/scripts/build.sh
# Model-export python deps.
pip install --upgrade ultralytics pillow numpy
ok "ExecuTorch + QNN backend built"

# =============================================================================
# 8. Stage the project's QNN scripts into ~/et_setup (LF copies the docs assume)
# =============================================================================
c "Staging scripts/qnn -> $ET_SETUP"
for f in export_yolo_backbone_qnn.py swap_model.sh deploy_default.sh start_npu_server.sh \
         build_socket_server.sh push_persistent.sh test_npu_client.py; do
  src="$REPO_ROOT/scripts/qnn/$f"
  if [ -f "$src" ]; then
    # normalize CRLF -> LF when copying to the build box
    sed 's/\r$//' "$src" > "$ET_SETUP/$f" && chmod +x "$ET_SETUP/$f"
  fi
done
ok "scripts staged"

# =============================================================================
# 9. Verify
# =============================================================================
c "Verify"
python - <<'PY'
import importlib, sys
mods = ["torch", "executorch", "ultralytics"]
for m in mods:
    try:
        importlib.import_module(m); print(f"  ✓ import {m}")
    except Exception as e:
        print(f"  ✗ import {m}: {e!r}"); sys.exit(1)
try:
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    assert hasattr(QcomChipset, "SM8750"), "QcomChipset.SM8750 missing"
    print("  ✓ QcomChipset.SM8750 resolves (QNN AOT ready)")
except Exception as e:
    print(f"  ✗ QNN backend check: {e!r}"); sys.exit(1)
PY

cat <<EOF

\033[1;32m========================================================================
 Build box ready.
========================================================================\033[0m

 Next:
   source ~/et_setup/env.sh
   MODEL_PT=<your.pt> OUT_PTE_BASE=<out> python scripts/qnn/export_yolo_backbone_qnn.py
   # then verify ~3.8 MB + 'strings <pte> | grep -c QnnBackend' >= 1
   # full recipe: MODEL_BUILD_GUIDE.md

 NOTE: the QNN model runs on-device in a shell-uid server (untrusted_app can't
 open the cDSP). Deploy a model bundle with:  scripts/qnn/swap_model.sh models/<name>
 First device bring-up also needs:            scripts/qnn/push_persistent.sh
EOF

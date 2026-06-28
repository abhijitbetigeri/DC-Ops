#!/usr/bin/env bash
# =============================================================================
# Run the DC-Ops build-box setup inside a Linux x86_64 Docker container.
#
# For machines that can't host the QNN AOT build natively — chiefly macOS
# (whose shell is Unix, not Linux, and can't load the SDK's linux ELF libs).
# Works on Intel Macs (native amd64) and Apple Silicon (Rosetta-emulated amd64,
# slower but the compile is CPU-bound so it's fine). Also works on any Linux host.
#
# What it does:
#   - starts an x86_64 ubuntu:22.04 container
#   - mounts the repo at /work  (so built .pte files land back on your machine)
#   - persists the heavy build ($HOME inside the container) in a named volume
#     'dcops-buildbox' so re-runs are incremental, not from scratch
#   - runs setup/setup.sh inside it
#
# QNN SDK (license-gated) — put the downloaded zip somewhere under the repo (or
# pass its path) and set QNN_SDK_ZIP to the path AS SEEN INSIDE THE CONTAINER,
# e.g. if it's at <repo>/setup/qairt.zip:
#     QNN_SDK_ZIP=/work/setup/qairt.zip bash setup/run-in-docker.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VOLUME="${DCOPS_VOLUME:-dcops-buildbox}"
IMAGE="${DCOPS_IMAGE:-ubuntu:22.04}"

command -v docker >/dev/null 2>&1 || {
  echo "Docker not found. Install Docker Desktop (macOS/Windows) or docker engine (Linux)." >&2
  echo "macOS: https://www.docker.com/products/docker-desktop/  (enable 'Use Rosetta' on Apple Silicon)" >&2
  exit 1
}

# Force amd64 so the QNN x86_64-linux libs run (emulated on Apple Silicon).
PLATFORM="${DCOPS_PLATFORM:-linux/amd64}"

echo "== launching $IMAGE ($PLATFORM); repo -> /work; build cache -> volume '$VOLUME' =="
exec docker run --rm -it \
  --platform "$PLATFORM" \
  -v "$REPO_ROOT":/work \
  -v "$VOLUME":/root \
  -e HOME=/root \
  ${QNN_SDK_ZIP:+-e QNN_SDK_ZIP="$QNN_SDK_ZIP"} \
  -w /work \
  "$IMAGE" \
  bash -lc 'apt-get update -y >/dev/null && apt-get install -y sudo >/dev/null 2>&1 || true; bash /work/setup/setup.sh'

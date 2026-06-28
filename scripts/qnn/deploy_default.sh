#!/bin/bash
# deploy_default.sh — one-touch deploy of the repo's DEFAULT model bundle
# (models/dcops-v1) to the on-device QNN NPU server. Just swap_model.sh aimed
# at the checked-in default bundle, for a fresh-device bring-up.
#
# IMPORTANT — why the .pte is deployed this way (and NOT bundled in the APK):
#   The NPU .pte runs inside the persistent shell-uid server in
#   /data/local/tmp/dcops_yolo_qnn. An untrusted_app (the installed APK,
#   com.dcops.ar.qnn) CANNOT write /data/local/tmp and CANNOT open an
#   unsigned-PD cDSP/HTP session on retail hardware. So the QNN model cannot
#   live in the APK's assets — the repo bundle (models/dcops-v1) PLUS these
#   scripts ARE the deployment path. (Only the CPU variant, android-app, bundles
#   its .pte in assets, because it runs the model in-process via XNNPACK.)
#
# Prereq for a truly fresh device: the server binary + QNN libs must be staged
# once via ~/et_setup/push_persistent.sh (swap_model.sh only pushes the model
# bundle: model.pte + meta.txt + labels.txt). swap_model.sh warns if the binary
# is missing.
set -euo pipefail

BUNDLE=/mnt/c/Users/Rashi/AndroidStudioProjects/DC-Ops/models/dcops-v1
echo "== deploying default bundle: $BUNDLE =="
bash "$HOME/et_setup/swap_model.sh" "$BUNDLE"

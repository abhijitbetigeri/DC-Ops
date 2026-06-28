#!/bin/bash
# swap_model.sh — deploy a model "bundle" to the on-device QNN NPU server.
#
# Usage:
#   swap_model.sh <bundle-dir | model.pte | hf-url> [model.json]
#
#   <bundle-dir>  a directory containing model.pte + model.json
#   <model.pte>   a lone .pte file; pass model.json as the 2nd arg (or it
#                 falls back to the repo default bundle's model.json)
#   <hf-url>      a URL to a .pte (downloaded with IPv4-forced retries); pass
#                 model.json as the 2nd arg (or fall back to the default)
#
# A bundle on the device is:  model.pte + meta.txt + labels.txt  in $WS.
# meta.txt / labels.txt are GENERATED here from the human-authored model.json
# (the C++ server reads the simple meta.txt/labels.txt, not json).
#
# Wire contract (see android-app-qnn + qnn_socket_server.cpp): the server
# derives input_w/h, num_classes, num_mask, strides, anchors from the .pte's
# ExecuTorch metadata and validates against meta.txt; reg_max comes from meta.
#
# NOTE: never use `pkill -f qnn_socket_server` (it self-matches this/any script
# whose command line contains that string -> exit 137). start_npu_server.sh
# kills the server by PID.
set -euo pipefail

SER=${SER:-R3CXC08005Y}
WS=/data/local/tmp/dcops_yolo_qnn
ETSETUP="$HOME/et_setup"
# Repo default bundle (so a lone .pte / url can borrow its model.json + labels).
DEFAULT_JSON=/mnt/c/Users/Rashi/AndroidStudioProjects/DC-Ops/models/dcops-v1/model.json

if [ $# -lt 1 ]; then
  echo "usage: swap_model.sh <bundle-dir | model.pte | hf-url> [model.json]" >&2
  exit 2
fi
SRC="$1"
JSON_ARG="${2:-}"

WORK="$(mktemp -d /tmp/swap_model.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
PTE=""
JSON=""

download_pte() {  # $1=url $2=dest
  python3 - "$1" "$2" <<'PY'
import sys, socket, urllib.request, time
socket.setdefaulttimeout(60)  # don't let one stalled connection hang forever
# WSL here has broken IPv6 -> force IPv4 so downloads are fast, not 30s SYN-SENT stalls.
_g = socket.getaddrinfo
socket.getaddrinfo = lambda host, *a, **k: [r for r in _g(host, *a, **k) if r[0] == socket.AF_INET]
url, out = sys.argv[1], sys.argv[2]
last = None
for i in range(5):
    try:
        urllib.request.urlretrieve(url, out)
        print(f"downloaded {out} ({__import__('os').path.getsize(out)} bytes)")
        break
    except Exception as e:
        last = e
        print(f"  retry {i+1}/5: {e!r}", flush=True)
        time.sleep(3 * (i + 1))
else:
    raise last
PY
}

# --- resolve source -> PTE + JSON --------------------------------------------
if [ -d "$SRC" ]; then
  PTE="$SRC/model.pte"
  JSON="${JSON_ARG:-$SRC/model.json}"
elif printf '%s' "$SRC" | grep -qiE '^https?://'; then
  echo "== downloading .pte from $SRC =="
  PTE="$WORK/model.pte"
  download_pte "$SRC" "$PTE"
  JSON="${JSON_ARG:-$DEFAULT_JSON}"
else
  PTE="$SRC"
  JSON="${JSON_ARG:-$DEFAULT_JSON}"
fi

[ -f "$PTE" ]  || { echo "ERROR: model.pte not found: $PTE" >&2; exit 1; }
[ -f "$JSON" ] || { echo "ERROR: model.json not found: $JSON (pass it as 2nd arg)" >&2; exit 1; }
echo "pte  = $PTE ($(stat -c%s "$PTE") bytes)"
echo "json = $JSON"

# --- generate meta.txt + labels.txt from model.json --------------------------
python3 - "$JSON" "$WORK/meta.txt" "$WORK/labels.txt" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as f:
    f.write(f"type={j.get('type','yolov8_seg')}\n")
    f.write(f"reg_max={j.get('reg_max',16)}\n")
    f.write(f"input_size={j.get('input_size',640)}\n")
labels = j.get("labels", [])
with open(sys.argv[3], "w") as f:
    for lab in labels:
        f.write(str(lab) + "\n")
print(f"meta.txt + labels.txt generated ({len(labels)} labels)")
PY
echo "--- meta.txt ---";   cat "$WORK/meta.txt"
echo "--- labels.txt ---"; cat "$WORK/labels.txt"

# --- push bundle to device ---------------------------------------------------
echo "== pushing bundle to $SER:$WS =="
adb -s "$SER" shell "mkdir -p $WS"
adb -s "$SER" push "$PTE"            "$WS/model.pte"
adb -s "$SER" push "$WORK/meta.txt"  "$WS/meta.txt"
adb -s "$SER" push "$WORK/labels.txt" "$WS/labels.txt"

# Warn if the server binary isn't staged yet (fresh device needs push_persistent.sh first).
if ! adb -s "$SER" shell "ls $WS/qnn_socket_server" >/dev/null 2>&1; then
  echo "WARNING: $WS/qnn_socket_server not found on device."
  echo "         Run ~/et_setup/push_persistent.sh first to stage the server + QNN libs."
fi

# --- restart the server + show warmup/model/derivation lines -----------------
echo "== restarting NPU server =="
bash "$ETSETUP/start_npu_server.sh"
echo "== server.log (warmup / model / derivation) =="
adb -s "$SER" shell "grep -iE 'warmup|listening|model|meta|label|deriv|num_classes|input|FATAL' $WS/server.log" || true

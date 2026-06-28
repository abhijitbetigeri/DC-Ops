#!/bin/bash
# Start (or restart) the PERSISTENT shell-context NPU server on the phone.
# The DC-Ops QNN app (com.dcops.ar.qnn) connects to it over 127.0.0.1:8765.
# Run this once after each device reboot (the daemon runs as the `shell` user,
# which is the only context allowed an unsigned-PD cDSP session on retail S25).
#
# qnn_socket_server loads the .pte ONCE (Module API) and runs forward() per frame
# (~3ms NPU), vs the old per-frame runner spawn (~190ms). Input is raw RGBA8888;
# the server does RGBA->NCHW float/255. Assumes artifacts staged in $WS by
# push_persistent.sh / deploy_persistent_server.sh.
SER=${SER:-R3CXC08005Y}
WS=/data/local/tmp/dcops_yolo_qnn

# Kill by PID (pkill -f would match this script's own command line -> self-kill).
PIDS=$(adb -s "$SER" shell "ps -A | grep qnn_socket_server | grep -v grep | awk '{print \$2}'")
[ -n "$PIDS" ] && adb -s "$SER" shell "kill -9 $PIDS" 2>/dev/null
sleep 1
# Detach fully so adb returns and the daemon survives the adb session.
adb -s "$SER" shell "cd $WS && setsid sh -c 'LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./qnn_socket_server model.pte >server.log 2>&1' </dev/null >/dev/null 2>&1 &" &
sleep 4
echo "=== server.log ==="
adb -s "$SER" shell "grep -E 'warmup|listening|FATAL' $WS/server.log"
echo "=== process ==="
adb -s "$SER" shell "ps -A | grep qnn_socket_server | grep -v grep || echo 'NOT RUNNING'"

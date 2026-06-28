#!/bin/bash
# Build the dependency-free NPU TCP relay, stage it + the runner + QNN libs in
# /data/local/tmp (DSP-accessible, shell-owned), and launch it as a persistent
# shell-uid daemon. The DC-Ops app then talks to it over 127.0.0.1:8765.
set -e
source "$HOME/et_setup/env.sh" >/dev/null 2>&1
SER=${SER:-R3CXC08005Y}
WS=/data/local/tmp/dcops_yolo_qnn
NDK=$HOME/et_setup/ndk/android-ndk-r26c
CLANG=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang
RUNNER=$(find $HOME/executorch/build-android -name qnn_executor_runner 2>/dev/null | head -1)
BACKEND=$(find $HOME/executorch/build-android -name libqnn_executorch_backend.so 2>/dev/null | head -1)
PTE=$HOME/executorch/dcops_qnn/dc_ops_yolov8n_seg_backbone_qnn.pte
QNN=$HOME/et_setup/qairt/2.46.0.260424

echo "== compiling server =="
"$CLANG" $HOME/et_setup/yolo_npu_server.c -o $HOME/et_setup/yolo_npu_server -O2 -Wall
echo "built: $(file $HOME/et_setup/yolo_npu_server | cut -d: -f2-)"

echo "== staging to $WS =="
adb -s $SER shell "mkdir -p $WS/out"
adb -s $SER push "$HOME/et_setup/yolo_npu_server" $WS/ >/dev/null
adb -s $SER push "$RUNNER" $WS/qnn_executor_runner >/dev/null
adb -s $SER push "$BACKEND" $WS/ >/dev/null
for f in libQnnHtp.so libQnnSystem.so libQnnHtpV79Stub.so libQnnHtpPrepare.so; do
  adb -s $SER push "$QNN/lib/aarch64-android/$f" $WS/ >/dev/null
done
adb -s $SER push "$QNN/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" $WS/ >/dev/null
adb -s $SER push "$PTE" $WS/model.pte >/dev/null
adb -s $SER shell "echo input.raw > $WS/input_list.txt; chmod +x $WS/qnn_executor_runner $WS/yolo_npu_server"

echo "== (re)launching daemon =="
adb -s $SER shell "pkill -f yolo_npu_server || true; sleep 1"
adb -s $SER shell "cd $WS && setsid sh -c './yolo_npu_server >server.log 2>&1' </dev/null >/dev/null 2>&1 &"
sleep 2
echo "== server.log =="
adb -s $SER shell "cat $WS/server.log; echo '---'; ps -A | grep yolo_npu_server | grep -v grep || echo 'NOT RUNNING'"

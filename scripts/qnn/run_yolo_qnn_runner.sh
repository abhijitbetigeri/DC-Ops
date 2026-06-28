#!/bin/bash
# Verify the YOLO QNN backbone .pte runs on the real Hexagon NPU via qnn_executor_runner
# in /data/local/tmp (DSP-accessible path) -- the same approach that worked for the chatbot.
source "$HOME/et_setup/env.sh" >/dev/null 2>&1
SER=R3CXC08005Y
WS=/data/local/tmp/dcops_yolo_qnn
PTE="$HOME/executorch/dcops_qnn/dc_ops_yolov8n_seg_backbone_qnn.pte"
RUNNER="$(find $HOME/executorch/build-android -name qnn_executor_runner 2>/dev/null | head -1)"
BACKEND="$(find $HOME/executorch/build-android -name libqnn_executorch_backend.so 2>/dev/null | head -1)"
QNN="$HOME/et_setup/qairt/2.46.0.260424"
echo "runner=$RUNNER"; echo "backend=$BACKEND"

# preprocess the test image -> NCHW float32 raw + input_list
python - <<'PY'
import numpy as np
from PIL import Image
im=Image.open("/mnt/c/Users/Rashi/AndroidStudioProjects/DC-Ops/android-app-qnn/app/src/main/assets/test_rack.jpg").convert("RGB").resize((640,640))
a=(np.asarray(im,dtype=np.float32)/255.).transpose(2,0,1)[None].copy()
a.tofile("/tmp/input.raw"); open("/tmp/input_list.txt","w").write("input.raw\n")
print("input", a.shape)
PY

adb -s $SER shell "rm -rf $WS; mkdir -p $WS/out"
adb -s $SER push "$PTE" $WS/model.pte >/dev/null
adb -s $SER push "$RUNNER" $WS/qnn_executor_runner >/dev/null
adb -s $SER push "$BACKEND" $WS/ >/dev/null
for f in libQnnHtp.so libQnnSystem.so libQnnHtpV79Stub.so libQnnHtpPrepare.so; do adb -s $SER push "$QNN/lib/aarch64-android/$f" $WS/ >/dev/null; done
adb -s $SER push "$QNN/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" $WS/ >/dev/null
adb -s $SER push /tmp/input.raw /tmp/input_list.txt $WS/ >/dev/null
echo "=== RUN on NPU ==="
adb -s $SER shell "cd $WS && chmod +x qnn_executor_runner && export LD_LIBRARY_PATH=. && export ADSP_LIBRARY_PATH=. && ./qnn_executor_runner --model_path model.pte --input_list input_list.txt --output_folder out 2>&1 | grep -viE 'INFO' | tail -25"
echo "=== output files ==="
adb -s $SER shell "ls -la $WS/out"

#!/bin/bash
# Benchmark ExecuTorch models on S25 Ultra via ADB
# Compares XNNPACK (CPU) vs QNN HTP (NPU) inference
#
# Usage: bash scripts/benchmark_on_device.sh
#
# Prerequisites:
#   - S25 connected via ADB
#   - Both .pte files pushed to /data/local/tmp/
#   - ExecuTorch executor_runner binary on device

ADB="adb"
DEVICE_DIR="/data/local/tmp/dc_ops"
RESULTS_DIR="benchmark_results"

mkdir -p $RESULTS_DIR

echo "=== DC-Ops Model Benchmark ==="
echo "Device: $($ADB shell getprop ro.product.model)"
echo "SoC: $($ADB shell getprop ro.soc.model)"
echo ""

# Push models if not already there
echo "Pushing models to device..."
$ADB shell mkdir -p $DEVICE_DIR
$ADB push models/dc_ops_yolov8n_seg.pte $DEVICE_DIR/xnnpack_model.pte 2>/dev/null
$ADB push models/dc_ops_retinanet_qnn.pte $DEVICE_DIR/qnn_model.pte 2>/dev/null

# Create a test input on device (random tensor, 1x3x640x640)
echo "Creating test input..."
$ADB shell "cd $DEVICE_DIR && dd if=/dev/urandom of=test_input.bin bs=1228800 count=1 2>/dev/null"

# Benchmark function
benchmark_model() {
    local model_name=$1
    local pte_file=$2
    local num_runs=${3:-20}

    echo ""
    echo "--- Benchmarking: $model_name ($num_runs runs) ---"

    # Get battery level before
    local batt_before=$($ADB shell dumpsys battery | grep level | awk '{print $2}')

    # Get CPU temp before
    local temp_before=$($ADB shell cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "N/A")

    # Run inference and capture timing
    # Using the ExecuTorch executor_runner if available, otherwise just measure push time
    if $ADB shell "test -f $DEVICE_DIR/executor_runner" 2>/dev/null; then
        $ADB shell "cd $DEVICE_DIR && ./executor_runner \
            --model_path $pte_file \
            --input_list_path test_input.bin \
            --iteration $num_runs \
            --warm_up 3" 2>&1 | tee $RESULTS_DIR/${model_name}_output.txt
    else
        echo "executor_runner not found on device."
        echo "To run full benchmark, build and push executor_runner."
        echo ""
        echo "Alternative: measure via Android app toggle."
    fi

    # Get battery level after
    local batt_after=$($ADB shell dumpsys battery | grep level | awk '{print $2}')

    # Get CPU temp after
    local temp_after=$($ADB shell cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "N/A")

    echo "Battery: $batt_before% -> $batt_after%"
    echo "Temp: $temp_before -> $temp_after"
}

# Run benchmarks
benchmark_model "XNNPACK_CPU" "xnnpack_model.pte" 20
benchmark_model "QNN_HTP_NPU" "qnn_model.pte" 20

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to $RESULTS_DIR/"

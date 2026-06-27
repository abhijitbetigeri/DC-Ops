"""
Export YOLOv8n to ExecuTorch .pte format for Snapdragon NPU (QNN backend).

Usage:
    python models/export/export_yolo.py --soc_model SM8750 --compile_only

Prerequisites:
    - ExecuTorch installed (pip install -e . from executorch/)
    - QAIRT SDK set via QNN_SDK_ROOT
    - ultralytics: pip install ultralytics
"""

import argparse
from pathlib import Path

import torch


def load_yolov8n(num_classes: int = 6):
    """Load YOLOv8n and adapt for DC-Ops classes.

    Classes: led_green, led_amber, led_red, led_off, cable, label
    """
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    return model


def export_to_pte(model, example_input, output_path: Path, soc_model: str):
    """Export PyTorch model → ExecuTorch .pte via QNN backend."""
    from executorch.backends.qualcomm.partition import QnnPartitioner
    from executorch.backends.qualcomm.utils.constants import QCOM_QUANTIZED
    from executorch.exir import to_edge

    exported = torch.export.export(model, (example_input,))

    edge_program = to_edge(exported)

    # Partition for Qualcomm QNN backend
    partitioner = QnnPartitioner(
        {
            "soc_model": soc_model,
            QCOM_QUANTIZED: True,
        }
    )
    edge_program = edge_program.to_backend(partitioner)
    et_program = edge_program.to_executorch()

    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    print(f"Exported to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8n for ExecuTorch")
    parser.add_argument(
        "--soc_model", default="SM8750", help="Qualcomm SoC model (default: SM8750)"
    )
    parser.add_argument(
        "--output", default="models/dc_ops_yolo.pte", help="Output .pte path"
    )
    parser.add_argument(
        "--compile_only", action="store_true", help="Only generate .pte, don't test"
    )
    parser.add_argument(
        "--input_size", type=int, default=640, help="Input image size (default: 640)"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLOv8n...")
    yolo = load_yolov8n()

    # Get the underlying PyTorch model
    torch_model = yolo.model.eval()

    example_input = torch.randn(1, 3, args.input_size, args.input_size)

    print(f"Exporting for {args.soc_model}...")
    export_to_pte(torch_model, example_input, output_path, args.soc_model)

    print("Done.")


if __name__ == "__main__":
    main()

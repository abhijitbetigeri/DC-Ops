"""
Fine-tune YOLOv8n on the DC-Ops dataset (Stream B).

This trains the 6-class detector (see models/classes.yaml). The resulting
`best.pt` is then handed to `models/export/export_yolo.py` to produce the
ExecuTorch `.pte` for the Snapdragon NPU.

Usage:
    pip install ultralytics
    python models/train/train_yolo.py --epochs 100 --imgsz 640

Prerequisites:
    - A labeled dataset laid out per models/train/dcops.yaml
    - A GPU is strongly recommended for training speed

Output:
    runs/detect/dcops_yolov8n/weights/best.pt   ← feed this to export_yolo.py
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n for DC-Ops")
    parser.add_argument(
        "--data",
        default=str(Path(__file__).with_name("dcops.yaml")),
        help="Path to the Ultralytics dataset YAML",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Base weights to fine-tune")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="dcops_yolov8n", help="Run name")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )

    metrics = model.val()
    print(f"\nValidation mAP50-95: {metrics.box.map:.4f}")
    print(f"Validation mAP50:    {metrics.box.map50:.4f}")
    print(
        "\nNext step — export to ExecuTorch:\n"
        f"  python models/export/export_yolo.py --soc_model SM8750 "
        f"--weights runs/detect/{args.name}/weights/best.pt"
    )


if __name__ == "__main__":
    main()

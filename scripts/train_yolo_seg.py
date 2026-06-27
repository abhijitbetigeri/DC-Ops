"""
Fine-tune YOLOv8n-seg on auto-labeled DC images.

Expects YOLO-seg format labels in data/labels/ (from auto_label.py).

Usage:
    python scripts/train_yolo_seg.py
    python scripts/train_yolo_seg.py --epochs 50 --batch 8
"""

import argparse
import shutil
from pathlib import Path

import yaml


def prepare_dataset(images_dir: Path, labels_dir: Path, output_dir: Path, split: float = 0.85):
    """Split images + labels into train/val sets in YOLO format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_img = output_dir / "train" / "images"
    train_lbl = output_dir / "train" / "labels"
    val_img = output_dir / "val" / "images"
    val_lbl = output_dir / "val" / "labels"

    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    ])

    labeled = []
    for img in image_files:
        label_file = labels_dir / f"{img.stem}.txt"
        if label_file.exists() and label_file.stat().st_size > 0:
            labeled.append((img, label_file))

    if not labeled:
        print("ERROR: No labeled images found. Run auto_label.py first.")
        return None

    split_idx = int(len(labeled) * split)
    train_set = labeled[:split_idx]
    val_set = labeled[split_idx:]

    for img, lbl in train_set:
        shutil.copy2(img, train_img / img.name)
        shutil.copy2(lbl, train_lbl / f"{img.stem}.txt")

    for img, lbl in val_set:
        shutil.copy2(img, val_img / img.name)
        shutil.copy2(lbl, f"{img.stem}.txt")

    print(f"Dataset: {len(train_set)} train, {len(val_set)} val")
    return output_dir


def create_dataset_yaml(dataset_dir: Path, classes: list[str]) -> Path:
    """Create YOLO dataset.yaml config."""
    config = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {i: name for i, name in enumerate(classes)},
    }

    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Dataset config: {yaml_path}")
    return yaml_path


DC_CLASSES = [
    "server rack",
    "compute tray",
    "NVLink switch tray",
    "network switch",
    "power shelf",
    "cable",
    "network port",
    "LED indicator",
    "label",
    "fan",
    "cooling manifold",
    "cable cartridge",
    "power connector",
    "drive bay",
    "management port",
    "DPU",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--images_dir", default="data/sample_images")
    parser.add_argument("--labels_dir", default="data/labels")
    parser.add_argument("--dataset_dir", default="data/dataset")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    dataset_dir = Path(args.dataset_dir)

    print("Preparing dataset...")
    result = prepare_dataset(images_dir, labels_dir, dataset_dir)
    if result is None:
        return

    yaml_path = create_dataset_yaml(dataset_dir, DC_CLASSES)

    print(f"\nStarting YOLOv8n-seg fine-tuning...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch:  {args.batch}")
    print(f"  ImgSz:  {args.imgsz}")

    from ultralytics import YOLO

    model = YOLO("yolov8n-seg.pt")

    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project="models/runs",
        name="dc_ops_seg",
        exist_ok=True,
        device="mps",  # Apple Silicon GPU; change to "cpu" if issues
        patience=10,
        save=True,
        plots=True,
    )

    best_weights = Path("models/runs/dc_ops_seg/weights/best.pt")
    if best_weights.exists():
        output = Path("models/dc_ops_yolov8n_seg.pt")
        shutil.copy2(best_weights, output)
        print(f"\nBest model saved to {output}")
        print(f"  Size: {output.stat().st_size / 1e6:.1f} MB")
    else:
        print("\nWARNING: best.pt not found. Check training logs.")

    print("\nNext step: export to ExecuTorch .pte")
    print("  python models/export/export_yolo.py --model models/dc_ops_yolov8n_seg.pt")


if __name__ == "__main__":
    main()

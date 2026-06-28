"""
Run trained RetinaNet on validation images and visualize detections.
Produces a grid of annotated images for the demo/slides.

Usage:
    python scripts/retinanet_demo.py
"""

import os
import random
import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

NUM_CLASSES = 16
CONF_THRESHOLD = 0.3

DC_CLASSES = [
    "server rack", "compute tray", "NVLink switch tray", "network switch",
    "power shelf", "cable", "network port", "LED indicator",
    "label", "fan", "cooling manifold", "cable cartridge",
    "power connector", "drive bay", "management port", "DPU",
]

COLORS = [
    (76, 175, 80), (255, 152, 0), (156, 39, 176), (33, 150, 243),
    (244, 67, 54), (0, 188, 212), (255, 235, 59), (0, 230, 118),
    (224, 64, 251), (121, 85, 72), (3, 169, 244), (255, 87, 34),
    (255, 193, 7), (139, 195, 74), (96, 125, 139), (255, 23, 68),
]


def load_model(weights_path):
    model = retinanet_resnet50_fpn_v2(weights=None)
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = NUM_CLASSES + 1
    in_channels = model.head.classification_head.cls_logits.in_channels
    model.head.classification_head.cls_logits = torch.nn.Conv2d(
        in_channels, num_anchors * (NUM_CLASSES + 1), kernel_size=3, stride=1, padding=1
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference(model, img_path, device):
    img = Image.open(img_path).convert("RGB")
    orig = img.copy()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640, 640)),
    ])
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(tensor)

    pred = preds[0]
    boxes = pred["boxes"].cpu()
    labels = pred["labels"].cpu()
    scores = pred["scores"].cpu()

    # scale boxes back to original image size (model ran at 640x640)
    ow, oh = orig.size
    sx, sy = ow / 640.0, oh / 640.0

    draw = ImageDraw.Draw(orig)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(14, ow // 50))
    except Exception:
        font = ImageFont.load_default()

    count = 0
    for box, label, score in zip(boxes, labels, scores):
        if score < CONF_THRESHOLD:
            continue
        cls_idx = label.item() - 1  # 1-indexed -> 0-indexed
        if cls_idx < 0 or cls_idx >= NUM_CLASSES:
            continue
        x1, y1, x2, y2 = box.tolist()
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
        color = COLORS[cls_idx % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{DC_CLASSES[cls_idx]} {score:.0%}"
        tb = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([tb[0], tb[1], tb[2] + 4, tb[3] + 2], fill=color)
        draw.text((x1 + 2, y1), text, fill=(0, 0, 0), font=font)
        count += 1

    return orig, count


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model("models/retinanet_dc_ops.pth")
    model.to(device)

    img_dir = Path("data/merged_dataset/val/images")
    out_dir = Path("data/retinanet_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(img_dir.glob("*.jpg"))
    # Pick a varied sample (ports + servers)
    random.seed(7)
    sample = random.sample(all_imgs, min(12, len(all_imgs)))

    print(f"Running inference on {len(sample)} images...")
    results = []
    for img_path in sample:
        annotated, count = run_inference(model, img_path, device)
        out_path = out_dir / f"det_{img_path.stem[:30]}.jpg"
        annotated.save(out_path)
        results.append((out_path.name, count))
        print(f"  {img_path.name[:40]}: {count} detections")

    total = sum(c for _, c in results)
    print(f"\nTotal: {total} detections across {len(sample)} images")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()

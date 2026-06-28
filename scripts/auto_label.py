"""
Auto-label DC images using Grounding DINO + SAM.

Grounding DINO detects DC components by text prompt → SAM generates masks →
output YOLO-seg format annotations for fine-tuning YOLOv8n-seg.

Usage:
    python scripts/auto_label.py
    python scripts/auto_label.py --visualize  # save annotated preview images
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import SamPredictor, sam_model_registry

# DC-specific classes for detection — granular for NVL72 and general DC infrastructure
DC_CLASSES = [
    "server rack",          # 0  - full rack enclosure / cabinet
    "compute tray",         # 1  - 1RU compute tray (GPU/CPU tray)
    "NVLink switch tray",   # 2  - 1RU NVLink switch tray
    "network switch",       # 3  - TOR switch, Ethernet/InfiniBand switch
    "power shelf",          # 4  - power supply shelf / PSU bank
    "cable",                # 5  - copper, fiber, power cables
    "network port",         # 6  - RJ45, OSFP, QSFP, Ethernet ports
    "LED indicator",        # 7  - status LEDs on trays/switches
    "label",                # 8  - serial number, asset tag, nameplate
    "fan",                  # 9  - cooling fans
    "cooling manifold",     # 10 - liquid cooling manifold/connection
    "cable cartridge",      # 11 - NVLink copper cable cartridge
    "power connector",      # 12 - power plug, bus bar connector
    "drive bay",            # 13 - NVMe drive bay / disk slot
    "management port",      # 14 - BMC RJ45, serial console port
    "DPU",                  # 15 - BlueField DPU / NIC card
]

CLASS_TO_ID = {name: i for i, name in enumerate(DC_CLASSES)}

GDINO_CONFIG = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS = "models/weights/groundingdino_swint_ogc.pth"
SAM_WEIGHTS = "models/weights/sam_vit_b_01ec64.pth"

INPUT_DIR = Path("data/sample_images")
OUTPUT_DIR = Path("data/labels")
VIS_DIR = Path("data/visualized")


def download_weights():
    """Download Grounding DINO and SAM weights if not present."""
    import urllib.request

    weights_dir = Path("models/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    gdino_path = Path(GDINO_WEIGHTS)
    if not gdino_path.exists():
        print("Downloading Grounding DINO weights (~700MB)...")
        urllib.request.urlretrieve(
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            gdino_path,
        )
        print("  Done.")

    sam_path = Path(SAM_WEIGHTS)
    if not sam_path.exists():
        print("Downloading SAM ViT-B weights (~375MB)...")
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            sam_path,
        )
        print("  Done.")


def setup_gdino_config():
    """Ensure Grounding DINO config file exists."""
    config_path = Path(GDINO_CONFIG)
    if not config_path.exists():
        import groundingdino
        pkg_dir = Path(groundingdino.__file__).parent
        possible = pkg_dir / "config" / "GroundingDINO_SwinT_OGC.py"
        if possible.exists():
            return str(possible)
        for p in pkg_dir.rglob("*.py"):
            if "GroundingDINO" in p.name and "SwinT" in p.name:
                return str(p)
    return GDINO_CONFIG


def mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    """Convert binary mask to normalized polygon points (YOLO-seg format)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return []

    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    h, w = mask.shape
    polygon = []
    for point in approx.squeeze():
        polygon.extend([float(point[0]) / w, float(point[1]) / h])

    return polygon


def label_image(
    image_path: Path,
    gdino_model,
    sam_predictor: SamPredictor,
    text_prompt: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.2,
    visualize: bool = False,
):
    """Run Grounding DINO + SAM on a single image, return YOLO-seg labels."""
    image_source, image_tensor = load_image(str(image_path))

    device = "cpu"
    boxes, logits, phrases = predict(
        model=gdino_model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    if len(boxes) == 0:
        return [], None

    h, w, _ = image_source.shape
    boxes_abs = boxes.clone()
    boxes_abs[:, [0, 2]] *= w
    boxes_abs[:, [1, 3]] *= h

    sam_predictor.set_image(image_source)

    labels = []
    vis_image = image_source.copy() if visualize else None

    for box, logit, phrase in zip(boxes_abs, logits, phrases):
        box_np = box.cpu().numpy()
        input_box = np.array([
            box_np[0] - box_np[2] / 2,
            box_np[1] - box_np[3] / 2,
            box_np[0] + box_np[2] / 2,
            box_np[1] + box_np[3] / 2,
        ])

        masks, scores, _ = sam_predictor.predict(
            box=input_box, multimask_output=False, return_logits=False
        )
        mask = masks[0]

        class_id = match_phrase_to_class(phrase)
        if class_id is None:
            continue

        polygon = mask_to_polygon(mask)
        if polygon:
            labels.append({"class_id": class_id, "polygon": polygon, "confidence": float(logit)})

        if visualize and vis_image is not None:
            color = [
                (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 128, 0), (128, 0, 128),
            ][class_id % 8]
            vis_image[mask] = (vis_image[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            x1, y1 = int(input_box[0]), int(input_box[1])
            cv2.putText(
                vis_image, f"{DC_CLASSES[class_id]} {logit:.2f}",
                (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

    return labels, vis_image


def match_phrase_to_class(phrase: str) -> int | None:
    """Map Grounding DINO detected phrase to our class ID."""
    phrase = phrase.lower().strip()
    # Order matters — more specific matches first
    mappings = [
        # 0 - server rack
        ("server rack", 0), ("rack", 0), ("cabinet", 0), ("enclosure", 0),
        # 1 - compute tray
        ("compute tray", 1), ("gpu tray", 1), ("cpu tray", 1), ("tray", 1),
        ("blade", 1), ("server", 1), ("chassis", 1), ("compute", 1),
        # 2 - NVLink switch tray
        ("nvlink switch", 2), ("nvswitch", 2), ("switch tray", 2),
        # 3 - network switch
        ("network switch", 3), ("ethernet switch", 3), ("tor switch", 3),
        ("infiniband", 3), ("switch", 3),
        # 4 - power shelf
        ("power shelf", 4), ("power supply", 4), ("psu", 4),
        ("power unit", 4),
        # 5 - cable
        ("cable", 5), ("wire", 5), ("cord", 5), ("fiber", 5), ("fibre", 5),
        # 6 - network port
        ("osfp", 6), ("qsfp", 6), ("ethernet port", 6), ("network port", 6),
        ("port", 6), ("socket", 6),
        # 7 - LED indicator
        ("led", 7), ("indicator", 7), ("status light", 7), ("light", 7),
        # 8 - label
        ("label", 8), ("tag", 8), ("sticker", 8), ("nameplate", 8),
        ("serial", 8), ("barcode", 8),
        # 9 - fan
        ("fan", 9), ("ventilation", 9), ("vent", 9), ("airflow", 9),
        # 10 - cooling manifold
        ("manifold", 10), ("coolant", 10), ("liquid cooling", 10),
        ("cooling", 10), ("cold plate", 10), ("heat sink", 10),
        # 11 - cable cartridge
        ("cable cartridge", 11), ("cartridge", 11),
        # 12 - power connector
        ("power connector", 12), ("power plug", 12), ("bus bar", 12),
        ("power cable", 12), ("power cord", 12),
        # 13 - drive bay
        ("drive bay", 13), ("drive slot", 13), ("disk", 13), ("nvme", 13),
        ("hard drive", 13), ("ssd", 13), ("drive", 13), ("storage", 13),
        # 14 - management port
        ("bmc", 14), ("management port", 14), ("serial console", 14),
        ("rj45", 14), ("management", 14), ("ipmi", 14),
        # 15 - DPU / NIC
        ("dpu", 15), ("bluefield", 15), ("nic", 15), ("network card", 15),
        ("connectx", 15), ("network interface", 15), ("adapter", 15),
    ]
    for key, class_id in mappings:
        if key in phrase:
            return class_id
    return None


def write_yolo_label(label_path: Path, labels: list[dict]):
    """Write labels in YOLO-seg format: class_id x1 y1 x2 y2 ... xn yn"""
    with open(label_path, "w") as f:
        for label in labels:
            coords = " ".join(f"{v:.6f}" for v in label["polygon"])
            f.write(f"{label['class_id']} {coords}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Save annotated previews")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.2)
    args = parser.parse_args()

    download_weights()

    config_path = setup_gdino_config()
    print(f"Loading Grounding DINO from {config_path}...")
    gdino_model = load_model(config_path, GDINO_WEIGHTS)

    print("Loading SAM ViT-B...")
    sam = sam_model_registry["vit_b"](checkpoint=SAM_WEIGHTS)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sam = sam.to(device)
    print(f"  SAM on device: {device}")
    sam_predictor = SamPredictor(sam)

    # Two passes — keeps prompt short enough for GDINO while covering all classes
    text_prompts = [
        "server rack. compute tray. server. network switch. cable. network port. LED light. label.",
        "fan. cooling manifold. power supply. drive bay. management port. cable cartridge. power connector. network card.",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        VIS_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    )
    print(f"\nProcessing {len(image_files)} images...")

    stats = {"total": 0, "labeled": 0, "objects": 0}

    for i, img_path in enumerate(image_files):
        stats["total"] += 1
        print(f"  [{i+1}/{len(image_files)}] {img_path.name}", end="")

        try:
            all_labels = []
            vis_image = None

            # Multi-pass: run each prompt group to detect all component types
            for prompt_idx, prompt in enumerate(text_prompts):
                labels, vis = label_image(
                    img_path, gdino_model, sam_predictor, prompt,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    visualize=(args.visualize and prompt_idx == 0),
                )
                all_labels.extend(labels)
                if vis is not None and vis_image is None:
                    vis_image = vis

            # Re-draw visualization with all labels from all passes
            if args.visualize and all_labels and vis_image is not None:
                image_source, _ = load_image(str(img_path))
                vis_image = image_source.copy()
                for label in all_labels:
                    class_id = label["class_id"]
                    color = [
                        (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 0),
                        (128, 0, 255), (0, 0, 255), (255, 255, 0), (255, 0, 255),
                        (0, 255, 255), (128, 128, 0), (0, 128, 128), (128, 0, 128),
                        (64, 255, 64), (255, 64, 64), (64, 64, 255), (192, 192, 0),
                    ][class_id % 16]
                    # Draw polygon
                    poly = label["polygon"]
                    h, w = vis_image.shape[:2]
                    pts = np.array(
                        [[int(poly[j] * w), int(poly[j+1] * h)] for j in range(0, len(poly), 2)],
                        dtype=np.int32,
                    )
                    cv2.polylines(vis_image, [pts], True, color, 2)
                    if len(pts) > 0:
                        cv2.putText(
                            vis_image, f"{DC_CLASSES[class_id]} {label['confidence']:.2f}",
                            (pts[0][0], max(pts[0][1] - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                        )

            if all_labels:
                label_path = OUTPUT_DIR / f"{img_path.stem}.txt"
                write_yolo_label(label_path, all_labels)
                stats["labeled"] += 1
                stats["objects"] += len(all_labels)
                print(f" → {len(all_labels)} objects")

                if args.visualize and vis_image is not None:
                    vis_path = VIS_DIR / f"{img_path.stem}_labeled.jpg"
                    cv2.imwrite(str(vis_path), vis_image)
            else:
                print(" → no detections")

        except Exception as e:
            print(f" → error: {e}")

    print(f"\nDone! {stats['labeled']}/{stats['total']} images labeled, {stats['objects']} total objects")
    print(f"Labels saved to {OUTPUT_DIR}/")

    classes_path = OUTPUT_DIR / "classes.txt"
    classes_path.write_text("\n".join(DC_CLASSES) + "\n")
    print(f"Class list saved to {classes_path}")

    meta = {
        "classes": DC_CLASSES,
        "class_to_id": CLASS_TO_ID,
        "stats": stats,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
    }
    meta_path = OUTPUT_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

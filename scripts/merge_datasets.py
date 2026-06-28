"""
Merge 4 Roboflow datasets into one unified YOLO dataset for DC-Ops.
Maps all source classes to our 16 DC-Ops classes.

Datasets:
  1. ports-yutnj (ym-pffnw) — 7 port classes
  2. server-detection (fujitsu) — 9 server/fiber classes
  3. server-vision (hunters) — 30 granular server component classes
  4. pc-ports (ports-gmmmp) — 4 port classes

Usage:
    python scripts/merge_datasets.py
"""

import os
import shutil
import random
from pathlib import Path
import yaml

DC_CLASSES = [
    "server rack",        # 0
    "compute tray",       # 1
    "NVLink switch tray", # 2
    "network switch",     # 3
    "power shelf",        # 4
    "cable",              # 5
    "network port",       # 6
    "LED indicator",      # 7
    "label",              # 8
    "fan",                # 9
    "cooling manifold",   # 10
    "cable cartridge",    # 11
    "power connector",    # 12
    "drive bay",          # 13
    "management port",    # 14
    "DPU",                # 15
]

# Ports ym (7 classes)
PORTS_YM_MAP = {
    0: 6,   # dp -> network port
    1: 6,   # ethernet -> network port
    2: 6,   # hdmi -> network port
    3: 12,  # power -> power connector
    4: 6,   # usb3a -> network port
    5: 6,   # usba -> network port
    6: 6,   # usbc -> network port
}

# Server Detection Fujitsu (9 classes)
FUJITSU_MAP = {
    0: 6,   # bulkhead -> network port
    1: 6,   # cfp2 -> network port
    2: 6,   # lc patch panel -> network port
    3: 6,   # lcbulkhead -> network port
    4: 6,   # patch module -> network port
    5: 15,  # sfp -> DPU
    6: 1,   # t200 -> compute tray
    7: 1,   # t310 -> compute tray
    8: 6,   # xfp -> network port
}

# Server Vision (30 classes)
SERVER_VISION_MAP = {
    0: 1,   # cmos_battery -> compute tray
    1: 13,  # cmos_battery_empty -> drive bay (empty slot)
    2: 1,   # cpu_socket_empty -> compute tray
    3: 1,   # cpu_socket_full -> compute tray
    4: 1,   # dimm_slot_empty -> compute tray
    5: 1,   # dimm_slot_full -> compute tray
    6: 13,  # disk_drive -> drive bay
    7: 13,  # drive_bay_empty -> drive bay
    8: 13,  # drive_bay_full -> drive bay
    9: 9,   # fan -> fan
    10: 13, # hdd_cage -> drive bay
    11: 6,  # header_port -> network port
    12: 6,  # header_port_full -> network port
    13: 10, # heatsink -> cooling manifold
    14: 14, # idsdm_connector -> management port
    15: 6,  # lan_port -> network port
    16: 14, # microsd_card_slot -> management port
    17: 15, # ndc_connector -> DPU
    18: 13, # nvme_slot -> drive bay
    19: 12, # pcie_power -> power connector
    20: 12, # pcie_power_full -> power connector
    21: 15, # pcie_slot -> DPU
    22: 4,  # power_supply -> power shelf
    23: 15, # riser_2_3_connector -> DPU
    24: 6,  # sata_sas_port -> network port
    25: 15, # sfp_port -> DPU
    26: 14, # tpm_connector -> management port
    27: 6,  # usb_2_0_connector -> network port
    28: 6,  # usb_c_port -> network port
    29: 6,  # vga -> network port
}

# PC Ports (4 classes)
PC_PORTS_MAP = {
    0: 6,   # hdmi -> network port
    1: 6,   # rj45 -> network port
    2: 6,   # usb-a -> network port
    3: 6,   # usb-c -> network port
}

DATASETS = [
    ("ports_ym", "data/roboflow_ports_ym", PORTS_YM_MAP),
    ("fujitsu", "data/roboflow_server_fujitsu", FUJITSU_MAP),
    ("server_vision", "data/roboflow_server_vision", SERVER_VISION_MAP),
    ("pc_ports", "data/roboflow_pc_ports", PC_PORTS_MAP),
]

OUTPUT_DIR = Path("data/merged_dataset")


def bbox_to_polygon(parts):
    cx, cy, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return f"{x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"


def remap_label_file(src_path, dst_path, class_map):
    lines = []
    with open(src_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            old_class = int(parts[0])
            new_class = class_map.get(old_class)
            if new_class is None:
                continue
            coords = bbox_to_polygon(parts[1:5])
            lines.append(f"{new_class} {coords}")

    if lines:
        with open(dst_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return True
    return False


def main():
    # Clean output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    total_train = 0
    total_val = 0

    for ds_name, ds_path, class_map in DATASETS:
        ds_path = Path(ds_path)
        if not ds_path.exists():
            print(f"  {ds_name}: NOT FOUND, skipping")
            continue

        for split_src, split_dst in [("train", "train"), ("valid", "val"), ("test", "train")]:
            img_dir = ds_path / split_src / "images"
            lbl_dir = ds_path / split_src / "labels"
            if not img_dir.exists():
                continue

            count = 0
            for img_file in sorted(img_dir.iterdir()):
                if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                    continue
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if not lbl_file.exists():
                    continue

                prefix = f"{ds_name}_{split_src}_"
                dst_img = OUTPUT_DIR / split_dst / "images" / f"{prefix}{img_file.name}"
                dst_lbl = OUTPUT_DIR / split_dst / "labels" / f"{prefix}{img_file.stem}.txt"

                shutil.copy2(img_file, dst_img)
                if remap_label_file(lbl_file, dst_lbl, class_map):
                    count += 1

            if split_dst == "train":
                total_train += count
            else:
                total_val += count

            if count > 0:
                print(f"  {ds_name}/{split_src}: {count} images -> {split_dst}")

    # Write dataset.yaml
    config = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {i: name for i, name in enumerate(DC_CLASSES)},
    }
    with open(OUTPUT_DIR / "dataset.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\nMerged dataset: {total_train} train, {total_val} val")
    print(f"Total: {total_train + total_val} images")
    print(f"Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

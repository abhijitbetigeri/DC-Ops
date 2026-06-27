# Data (Stream D)

Server-rack images + YOLO labels that feed Stream B's training.

## Layout (YOLO format)
```
data/
├── images/
│   ├── train/   *.jpg|*.png
│   └── val/     *.jpg|*.png
└── labels/
    ├── train/   *.txt   (one per image, same basename)
    └── val/     *.txt
```

Each label `.txt` has one line per object:
```
<class_id> <cx> <cy> <w> <h>      # all normalized 0..1, box center + size
```

## Classes (FROZEN — models/classes.yaml)
| id | name | what to label |
|----|------|---------------|
| 0 | led_green | a green status LED |
| 1 | led_amber | an amber/yellow status LED |
| 2 | led_red | a red/fault LED |
| 3 | led_off | an unlit LED |
| 4 | cable | a network/power cable (or empty port for "missing") |
| 5 | label | a nameplate / serial-number / asset-tag region |

## Labeling tips
- Tools: [Roboflow](https://roboflow.com), [labelImg](https://github.com/HumanSignal/labelImg), or CVAT.
- Aim for varied lighting/angles — racks are dim and reflective.
- ~150–300 labeled images is a reasonable hackathon target for a usable model.
- Keep a held-out `val/` split (~15–20%).

> Images are gitignored by default (large/sensitive). Share the dataset out-of-band.

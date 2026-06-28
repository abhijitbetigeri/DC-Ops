# NVL72 3D Model Dimension Notes

This model keeps reusable geometry in scene units, but physical connector and cable
dimensions are stored in millimeters in `src/config/physicalDimensions.js`.

## Calibrated Dimensions

- OSFP 800G module: 22.58 mm wide, 13 mm high, up to 116 mm deep with pull tab.
- NVIDIA-style 800G optical patching: modeled as separate MPO-12/APC fiber patch
  cables using a 3 mm jacket outside diameter and 30 mm minimum bend radius.
- 800G OSFP active copper profiles are included for reuse:
  - 30AWG ACC: 7.2 mm OD, 72 mm minimum bend radius.
  - 26AWG ACC: 8.9 mm OD, 89 mm minimum bend radius.
  - 28AWG DAC: 10.2 mm OD, 81 mm minimum bend radius.
  - 25AWG DAC: 12.1 mm OD, 86 mm minimum bend radius.
- E1.S slot envelope: front slot based on 9.5 mm enclosure thickness and
  33.75 mm enclosure width.
- USB-C opening: 8.34 mm by 2.56 mm.
- RJ45 management jack: modeled from common 8P8C jack/interface dimensions.

## Public Data Limitations

The exact NVL72 external cable part numbers, harness bend forms, retention clips,
and rack cable-management dimensions are not publicly specified. The model therefore
uses public OSFP MSA/NVIDIA LinkX-style dimensions and datasheet cable OD/bend-radius
profiles rather than proprietary vendor drawings.

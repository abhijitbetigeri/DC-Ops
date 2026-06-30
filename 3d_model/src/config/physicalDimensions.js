// Real-world dimensions are stored in millimeters and converted into the
// existing scene scale from the OSFP MSA module width.
export const SCENE_UNITS_PER_MM = 0.17 / 22.58;

export function mm(value) {
  return value * SCENE_UNITS_PER_MM;
}

export const PHYSICAL_DIMENSIONS_MM = {
  osfp: {
    moduleWidth: 22.58,
    moduleHeight: 13,
    moduleLengthWithPullTab: 116,
    visibleCageDepth: 12,
    frontPlugProtrusion: 16,
    strainReliefLength: 32,
    cageWall: 1,
    mouthWidth: 19.8,
    mouthHeight: 8.1,
    heatSinkFinPitch: 2.2,
    ledDiameter: 1.6
  },
  cables: {
    // NVIDIA 800G twin-port OSFP optics use separate MPO-12/APC fiber cables.
    // Comparable 800G OSFP optical cable datasheets list 3 mm OD and 30 mm MBR.
    nvidiaMpo12FiberPatch: {
      label: 'NVIDIA MPO-12/APC fiber patch',
      diameter: 3,
      minBendRadius: 30,
      connectorExitStraight: 42,
      material: 'fiberCable'
    },
    // 800G active copper / ACC datasheets commonly list 30AWG OD 7.2 mm,
    // 26AWG OD 8.9 mm, with MBR = 10x OD near the connector.
    osfpActiveCopper30Awg: {
      label: '800G OSFP active copper 30AWG',
      diameter: 7.2,
      minBendRadius: 72,
      connectorExitStraight: 135,
      material: 'dacCable'
    },
    osfpActiveCopper26Awg: {
      label: '800G OSFP active copper 26AWG',
      diameter: 8.9,
      minBendRadius: 89,
      connectorExitStraight: 156,
      material: 'dacCable'
    },
    // Hairtail+ OSFP DAC examples list 28AWG OD 10.2 mm and 26/25AWG OD 12.1 mm.
    osfpHairtail28Awg: {
      label: '800G OSFP DAC 28AWG',
      diameter: 10.2,
      minBendRadius: 81,
      connectorExitStraight: 81,
      material: 'dacCable'
    },
    osfpHairtail25Awg: {
      label: '800G OSFP DAC 25AWG',
      diameter: 12.1,
      minBendRadius: 86,
      connectorExitStraight: 86,
      material: 'dacCable'
    }
  },
  e1s: {
    frontSlotWidth: 9.5,
    frontSlotHeight: 33.75,
    carrierLipHeight: 3.2,
    latchHeight: 3.6,
    screwDiameter: 2.2
  },
  rj45: {
    jackWidth: 11.7,
    jackHeight: 8.2,
    shellWidth: 14.4,
    shellHeight: 11,
    shellDepth: 8.5
  },
  usbC: {
    openingWidth: 8.34,
    openingHeight: 2.56,
    shellWidth: 10.4,
    shellHeight: 4.6,
    tongueWidth: 6.8,
    tongueHeight: 0.65
  },
  sfp: {
    cageWidth: 13.4,
    cageHeight: 8.5,
    cageDepth: 8
  }
};

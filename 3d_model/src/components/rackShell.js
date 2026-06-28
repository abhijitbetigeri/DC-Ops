import * as THREE from '../vendor/three.js';
import { addBox, addCylinder } from '../geometry/primitives.js';

export function createRackShell({ materials, rack }) {
  const group = new THREE.Group();
  group.name = 'single-wide MGX rack shell';

  addBox(group, 'left side outer cabinet panel', { x: -rack.width / 2, y: 0, z: 0 }, { x: 0.18, y: rack.height, z: rack.depth }, materials.rackPaint);
  addBox(group, 'right side outer cabinet panel', { x: rack.width / 2, y: 0, z: 0 }, { x: 0.18, y: rack.height, z: rack.depth }, materials.rackPaint);
  addBox(group, 'top cabinet cap', { x: 0, y: rack.height / 2, z: 0 }, { x: rack.width, y: 0.18, z: rack.depth }, materials.rackPaint);
  addBox(group, 'bottom cabinet plinth', { x: 0, y: -rack.height / 2, z: 0 }, { x: rack.width, y: 0.24, z: rack.depth }, materials.rackPaint);
  addBox(group, 'rear black cable spine plane', { x: 0, y: 0, z: -rack.depth / 2 }, { x: rack.innerWidth, y: rack.innerHeight, z: 0.08 }, materials.innerBlack);

  addBox(group, 'left front vertical rail', { x: -rack.innerWidth / 2 - rack.railWidth / 2, y: 0, z: rack.frontZ + 0.03 }, { x: rack.railWidth, y: rack.innerHeight, z: 0.12 }, materials.rackEdge);
  addBox(group, 'right front vertical rail', { x: rack.innerWidth / 2 + rack.railWidth / 2, y: 0, z: rack.frontZ + 0.03 }, { x: rack.railWidth, y: rack.innerHeight, z: 0.12 }, materials.rackEdge);
  addBox(group, 'champagne left inner rack frame', { x: -1.54, y: 0, z: rack.frontZ + 0.085 }, { x: 0.055, y: rack.innerHeight * 0.82, z: 0.045 }, materials.champagneTrim);
  addBox(group, 'champagne right inner rack frame', { x: 1.54, y: 0, z: rack.frontZ + 0.085 }, { x: 0.055, y: rack.innerHeight * 0.82, z: 0.045 }, materials.champagneTrim);
  addBox(group, 'champagne top inner rack frame', { x: 0, y: rack.innerHeight * 0.41, z: rack.frontZ + 0.085 }, { x: 3.12, y: 0.06, z: 0.045 }, materials.champagneTrim);
  addBox(group, 'champagne bottom inner rack frame', { x: 0, y: -rack.innerHeight * 0.41, z: rack.frontZ + 0.085 }, { x: 3.12, y: 0.06, z: 0.045 }, materials.champagneTrim);
  addBox(group, 'left internal cable manager', { x: -1.94, y: 0, z: rack.frontZ + 0.12 }, { x: 0.12, y: rack.innerHeight * 0.92, z: 0.18 }, materials.trayDark);
  addBox(group, 'right internal cable manager', { x: 1.94, y: 0, z: rack.frontZ + 0.12 }, { x: 0.12, y: rack.innerHeight * 0.92, z: 0.18 }, materials.trayDark);
  addBox(group, 'front center recessed black shadow cavity', { x: 0, y: 0, z: rack.frontZ - 0.035 }, { x: 3.02, y: rack.innerHeight * 0.8, z: 0.035 }, materials.innerBlack);

  for (let i = 0; i < 32; i += 1) {
    const y = -rack.innerHeight / 2 + 0.24 + i * (rack.innerHeight - 0.48) / 31;
    addCylinder(group, 'front rail rack screw', { x: -1.79, y, z: rack.frontZ + 0.105 }, 0.012, 0.006, materials.osfpMetal, { segments: 14 });
    addCylinder(group, 'front rail rack screw', { x: 1.79, y, z: rack.frontZ + 0.105 }, 0.012, 0.006, materials.osfpMetal, { segments: 14 });
  }

  addBox(group, 'right 50V busbar cover', { x: 2, y: 0, z: 0.35 }, { x: 0.075, y: rack.innerHeight * 0.9, z: 0.08 }, materials.copper);
  addBox(group, 'left shadowed cabinet wall', { x: -1.93, y: 0, z: 0 }, { x: 0.05, y: rack.innerHeight, z: rack.depth * 0.9 }, materials.innerBlack);

  for (let i = 0; i < 5; i += 1) {
    addBox(group, 'side panel vertical service vent', { x: -2.06, y: 3.6 - i * 0.16, z: 0.72 }, { x: 0.012, y: 0.07, z: 0.18 }, materials.blackAnodized);
  }

  return group;
}

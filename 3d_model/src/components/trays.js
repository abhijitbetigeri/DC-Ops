import * as THREE from '../vendor/three.js';
import { addBox, addCylinder } from '../geometry/primitives.js';
import {
  createE1SSlot,
  createMiniSfpCage,
  createOSFPCage,
  createRJ45Port,
  createUSBCPort
} from './connectors.js';

export function createComputeTray({ materials, rack, y, height, index = 0, portRegistry }) {
  const z = rack.frontZ;
  const group = new THREE.Group();
  group.name = `compute tray ${index + 1}`;

  addBox(group, 'compute tray champagne face rail', { x: 0, y, z }, { x: 3.08, y: height * 0.86, z: 0.052 }, materials.champagneTrim);
  addBox(group, 'compute tray black module field', { x: 0, y, z: z + 0.034 }, { x: 2.84, y: height * 0.66, z: 0.03 }, materials.blackAnodized);
  addBox(group, 'compute tray upper separator shadow', { x: 0, y: y + height * 0.43, z: z + 0.045 }, { x: 3.04, y: 0.012, z: 0.025 }, materials.innerBlack);
  addBox(group, 'compute tray lower separator shadow', { x: 0, y: y - height * 0.43, z: z + 0.045 }, { x: 3.04, y: 0.012, z: 0.025 }, materials.innerBlack);
  addBox(group, 'compute tray dark service center', { x: 0, y, z: z + 0.055 }, { x: 0.58, y: height * 0.6, z: 0.03 }, materials.trayDark);

  [-1.02, 1.02].forEach((sideX, sideIndex) => {
    const direction = sideIndex === 0 ? -1 : 1;
    addBox(group, 'Orchid module faceplate bank', { x: sideX, y, z: z + 0.052 }, { x: 1.05, y: height * 0.58, z: 0.026 }, materials.trayFace);

    [y + height * 0.18, y - height * 0.18].forEach((rowY) => {
      group.add(createOSFPCage({
        materials,
        position: { x: sideX - direction * 0.12, y: rowY, z: z + 0.078 },
        scale: 0.82,
        name: 'compute tray OSFP cage',
        portRegistry
      }));
      group.add(createOSFPCage({
        materials,
        position: { x: sideX + direction * 0.06, y: rowY, z: z + 0.078 },
        scale: 0.82,
        name: 'compute tray OSFP cage',
        portRegistry
      }));
      group.add(createE1SSlot({
        materials,
        position: { x: sideX + direction * 0.28, y: rowY, z: z + 0.086 },
        scale: 0.72
      }));
    });
  });

  group.add(createOSFPCage({
    materials,
    position: { x: -0.18, y: y + height * 0.18, z: z + 0.084 },
    scale: 0.7,
    name: 'BlueField-4 frontend OSFP cage'
  }));
  group.add(createOSFPCage({
    materials,
    position: { x: 0, y: y + height * 0.18, z: z + 0.084 },
    scale: 0.7,
    name: 'BlueField-4 frontend OSFP cage',
    portRegistry: index % 3 === 0 ? portRegistry : null
  }));
  group.add(createRJ45Port({ materials, position: { x: 0.23, y: y + height * 0.14, z: z + 0.088 }, scale: 0.62 }));
  group.add(createUSBCPort({ materials, position: { x: -0.24, y: y - height * 0.16, z: z + 0.102 }, scale: 0.72 }));
  group.add(createMiniSfpCage({ materials, position: { x: -0.06, y: y - height * 0.16, z: z + 0.092 }, scale: 0.72 }));
  group.add(createMiniSfpCage({ materials, position: { x: 0.09, y: y - height * 0.16, z: z + 0.092 }, scale: 0.72 }));

  addBox(group, 'blind-mate service alignment handle', { x: 0.27, y: y - height * 0.16, z: z + 0.09 }, { x: 0.14, y: 0.033, z: 0.025 }, materials.osfpMetal);
  addCylinder(group, 'compute tray green status LED', { x: 0.42, y, z: z + 0.102 }, 0.011, 0.005, materials.ledGreen, { segments: 14 });
  addCylinder(group, 'compute tray amber service LED', { x: 0.48, y, z: z + 0.102 }, 0.007, 0.005, materials.ledAmber, { segments: 14 });
  addCylinder(group, 'left tray thumbscrew', { x: -1.6, y, z: z + 0.06 }, 0.019, 0.008, materials.osfpMetal, { segments: 18 });
  addCylinder(group, 'right tray thumbscrew', { x: 1.6, y, z: z + 0.06 }, 0.019, 0.008, materials.osfpMetal, { segments: 18 });

  return group;
}

export function createSwitchTray({ materials, rack, y, height, index = 0, portRegistry }) {
  const z = rack.frontZ;
  const group = new THREE.Group();
  group.name = `NVLink switch tray ${index + 1}`;

  addBox(group, 'NVLink switch blind-mate champagne faceplate', { x: 0, y, z }, { x: 3.08, y: height * 0.82, z: 0.055 }, materials.champagneTrim);
  addBox(group, 'NVLink switch dark service band', { x: 0, y, z: z + 0.04 }, { x: 2.78, y: height * 0.56, z: 0.028 }, materials.blackAnodized);
  addBox(group, 'NVLink switch center rear-cable spine cover', { x: 0, y, z: z + 0.068 }, { x: 0.18, y: height * 0.62, z: 0.03 }, materials.innerBlack);

  for (let col = 0; col < 14; col += 1) {
    const x = -1.22 + col * (2.44 / 13);
    addBox(group, 'NVLink switch service grille slit', { x, y, z: z + 0.072 }, { x: 0.09, y: height * 0.18, z: 0.01 }, materials.perforatedBlack, { castShadow: false });
  }

  if (index % 3 === 1) {
    group.add(createRJ45Port({ materials, position: { x: 1.33, y, z: z + 0.086 }, scale: 0.42 }));
  }
  addCylinder(group, 'switch tray status blue', { x: -1.6, y, z: z + 0.077 }, 0.012, 0.005, materials.ledBlue, { segments: 14 });
  addCylinder(group, 'switch tray green health LED', { x: -1.52, y, z: z + 0.077 }, 0.01, 0.005, materials.ledGreen, { segments: 14 });

  return group;
}

export function createPowerShelf({ materials, rack, y, height, index = 0 }) {
  const z = rack.frontZ;
  const group = new THREE.Group();
  group.name = `power shelf ${index + 1}`;

  addBox(group, 'power shelf dark outer face no front cabling', { x: 0, y, z }, { x: 3.08, y: height * 0.84, z: 0.06 }, materials.blackAnodized);
  addBox(group, 'power shelf champagne shelf rail', { x: 0, y: y + height * 0.39, z: z + 0.055 }, { x: 3.04, y: 0.025, z: 0.026 }, materials.champagneTrim);
  addBox(group, 'power shelf lower champagne shelf rail', { x: 0, y: y - height * 0.39, z: z + 0.055 }, { x: 3.04, y: 0.025, z: 0.026 }, materials.champagneTrim);

  for (let i = 0; i < 6; i += 1) {
    const x = -1.2 + i * 0.48;
    addBox(group, '18.3kW PSU rectifier dark grille', { x, y, z: z + 0.07 }, { x: 0.38, y: height * 0.56, z: 0.035 }, materials.perforatedBlack);
    addBox(group, 'PSU silver service handle', { x, y: y - height * 0.22, z: z + 0.095 }, { x: 0.24, y: height * 0.035, z: 0.025 }, materials.osfpMetal);

    for (let row = 0; row < 4; row += 1) {
      for (let slot = 0; slot < 4; slot += 1) {
        addBox(group, 'PSU perforation slot', { x: x - 0.11 + slot * 0.07, y: y + height * (-0.13 + row * 0.085), z: z + 0.1 }, { x: 0.035, y: height * 0.018, z: 0.012 }, materials.trayFace, { castShadow: false });
      }
    }
  }

  addCylinder(group, 'power shelf blue ID LED', { x: -1.58, y: y + height * 0.27, z: z + 0.085 }, 0.018, 0.005, materials.ledBlue, { segments: 16 });
  addCylinder(group, 'power shelf green power LED', { x: -1.48, y: y + height * 0.27, z: z + 0.085 }, 0.018, 0.005, materials.ledGreen, { segments: 16 });

  return group;
}

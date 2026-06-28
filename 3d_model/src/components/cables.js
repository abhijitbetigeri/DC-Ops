import * as THREE from '../vendor/three.js';
import { PHYSICAL_DIMENSIONS_MM, mm } from '../config/physicalDimensions.js';
import { addBox, addCylinder } from '../geometry/primitives.js';

export function createCableProfile(profileName = 'nvidiaMpo12FiberPatch') {
  return PHYSICAL_DIMENSIONS_MM.cables[profileName] ?? PHYSICAL_DIMENSIONS_MM.cables.nvidiaMpo12FiberPatch;
}

export function createOSFPCableEnd({ materials, start, profile, scale = 1 }) {
  const group = new THREE.Group();
  group.name = `${profile.label} OSFP cable end`;
  group.position.set(start.x, start.y, start.z);

  const osfp = PHYSICAL_DIMENSIONS_MM.osfp;
  const plugW = mm(osfp.moduleWidth) * scale;
  const plugH = mm(osfp.moduleHeight) * scale;
  const plugDepth = mm(osfp.frontPlugProtrusion) * scale;
  const insertedDepth = mm(osfp.visibleCageDepth * 0.65) * scale;
  const reliefLength = mm(osfp.strainReliefLength) * scale;
  const reliefRadius = mm(profile.diameter * 0.72) * scale;

  addBox(
    group,
    'OSFP plug body inserted through cage opening',
    { x: 0, y: 0, z: (plugDepth - insertedDepth) / 2 },
    { x: plugW * 0.94, y: plugH * 0.86, z: plugDepth + insertedDepth },
    materials.osfpMetal
  );
  addBox(group, 'OSFP pull tab latch', { x: plugW * 0.34, y: -plugH * 0.18, z: plugDepth + mm(1.2) * scale }, { x: plugW * 0.18, y: plugH * 0.22, z: mm(2.2) * scale }, materials.portVoid);
  addCylinder(group, 'molded OSFP strain relief boot', { x: 0, y: 0, z: plugDepth + reliefLength / 2 }, reliefRadius, reliefLength, materials.strainRelief, { segments: 18 });

  return group;
}

export function createPatchCable({ materials, rack, start, side, variant = 0, profileName = 'nvidiaMpo12FiberPatch' }) {
  const profile = createCableProfile(profileName);
  const sideX = side < 0 ? -1.83 : 1.83;
  const cableRadius = mm(profile.diameter / 2);
  const bendRadius = mm(profile.minBendRadius);
  const exitStraight = mm(profile.connectorExitStraight);
  const osfp = PHYSICAL_DIMENSIONS_MM.osfp;
  const plugDepth = mm(osfp.frontPlugProtrusion) * (start.scale ?? 1);
  const reliefLength = mm(osfp.strainReliefLength) * (start.scale ?? 1);
  const offset = variant * cableRadius * 1.8;
  const drop = Math.max(bendRadius * 1.6, 0.22) + (variant % 5) * cableRadius * 8;
  const endY = THREE.MathUtils.clamp(start.y - drop + (variant % 2) * 0.18, -4, 4);
  const zFront = rack.frontZ + Math.max(bendRadius * 2.2, 0.26) + offset;
  const startZ = start.z + Math.max(exitStraight, plugDepth + reliefLength);

  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(start.x, start.y, startZ),
    new THREE.Vector3(start.x + side * bendRadius, start.y + cableRadius, zFront),
    new THREE.Vector3(sideX, (start.y + endY) * 0.5, zFront + bendRadius),
    new THREE.Vector3(sideX, endY, zFront),
    new THREE.Vector3(sideX - side * bendRadius * 0.75, endY - cableRadius * 4, zFront - bendRadius * 0.7)
  ]);

  const geometry = new THREE.TubeGeometry(curve, 36, cableRadius, 10, false);
  const mesh = new THREE.Mesh(geometry, materials[profile.material] ?? materials.cable);
  mesh.name = `${profile.label} ${profile.diameter}mm OD patch cable`;
  mesh.castShadow = true;

  const group = new THREE.Group();
  group.name = mesh.name;
  group.add(createOSFPCableEnd({ materials, start, profile, scale: start.scale ?? 1 }));
  group.add(mesh);
  return group;
}

export function createCableSet({ materials, rack, portRegistry, profileName = 'nvidiaMpo12FiberPatch' }) {
  const group = new THREE.Group();
  group.name = 'front patch cable set';

  portRegistry.forEach((port, index) => {
    if (port.kind !== 'osfp') return;
    if (index % 2 === 0 || port.scale < 0.6) {
      group.add(createPatchCable({
        materials,
        rack,
        start: port,
        side: port.x < 0 ? -1 : 1,
        variant: index % 9,
        profileName
      }));
    }
  });

  return group;
}

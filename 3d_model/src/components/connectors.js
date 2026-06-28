import * as THREE from '../vendor/three.js';
import { PHYSICAL_DIMENSIONS_MM, mm } from '../config/physicalDimensions.js';
import { addBox, addCylinder, createRoundedRectShape } from '../geometry/primitives.js';

function maybeRegisterPort(registry, port) {
  if (registry) {
    registry.push(port);
  }
}

function addRectangularPortFrame(group, name, { width, height, openingWidth, openingHeight, depth, material }) {
  const sideWall = Math.max((width - openingWidth) / 2, 0.001);
  const topWall = Math.max((height - openingHeight) / 2, 0.001);

  addBox(group, `${name} upper wall`, { x: 0, y: height / 2 - topWall / 2, z: 0 }, { x: width, y: topWall, z: depth }, material);
  addBox(group, `${name} lower wall`, { x: 0, y: -height / 2 + topWall / 2, z: 0 }, { x: width, y: topWall, z: depth }, material);
  addBox(group, `${name} left wall`, { x: -width / 2 + sideWall / 2, y: 0, z: 0 }, { x: sideWall, y: openingHeight, z: depth }, material);
  addBox(group, `${name} right wall`, { x: width / 2 - sideWall / 2, y: 0, z: 0 }, { x: sideWall, y: openingHeight, z: depth }, material);
}

function addRoundedPortFrame(group, name, position, outerWidth, outerHeight, innerWidth, innerHeight, radius, material) {
  const shape = createRoundedRectShape(outerWidth, outerHeight, radius);
  shape.holes.push(createRoundedRectShape(innerWidth, innerHeight, Math.min(radius * 0.6, innerHeight * 0.45)));

  const mesh = new THREE.Mesh(new THREE.ShapeGeometry(shape), material);
  mesh.name = name;
  mesh.position.set(position.x, position.y, position.z);
  group.add(mesh);
  return mesh;
}

export function createOSFPCage({
  materials,
  position = { x: 0, y: 0, z: 0 },
  scale = 1,
  name = 'OSFP cage',
  portRegistry = null
}) {
  const group = new THREE.Group();
  group.name = name;
  group.position.set(position.x, position.y, position.z);

  const osfp = PHYSICAL_DIMENSIONS_MM.osfp;
  const w = mm(osfp.moduleWidth) * scale;
  const h = mm(osfp.moduleHeight) * scale;
  const d = mm(osfp.visibleCageDepth) * scale;
  const mouthW = mm(osfp.mouthWidth) * scale;
  const mouthH = mm(osfp.mouthHeight) * scale;
  const cageWall = mm(osfp.cageWall) * scale;
  const ledRadius = mm(osfp.ledDiameter / 2) * scale;

  addRectangularPortFrame(group, 'OSFP open stamped cage', {
    width: w,
    height: h,
    openingWidth: mouthW,
    openingHeight: mouthH,
    depth: d,
    material: materials.osfpMetal
  });
  addBox(group, 'OSFP hollow interior shadow at rear of cage', { x: 0, y: 0, z: -d * 0.22 }, { x: mouthW * 0.96, y: mouthH * 0.9, z: d * 0.5 }, materials.portVoid);
  addBox(group, 'upper cage rail', { x: 0, y: h / 2 - cageWall, z: d / 2 + cageWall * 0.8 }, { x: w * 0.9, y: cageWall, z: cageWall * 1.4 }, materials.rackEdge);
  addBox(group, 'lower cage rail', { x: 0, y: -h / 2 + cageWall, z: d / 2 + cageWall * 0.8 }, { x: w * 0.9, y: cageWall, z: cageWall * 1.4 }, materials.rackEdge);
  addBox(group, 'center latch spring', { x: 0, y: 0, z: d / 2 + cageWall }, { x: cageWall * 1.6, y: mouthH * 0.72, z: cageWall * 1.4 }, materials.trayDark);
  addBox(group, 'pull-tab latch recess', { x: w * 0.36, y: -h * 0.02, z: d / 2 + cageWall * 1.2 }, { x: w * 0.08, y: mouthH * 0.55, z: cageWall * 1.2 }, materials.portVoid);

  for (let i = -3; i <= 3; i += 1) {
    addBox(group, 'OSFP heat sink fin', { x: i * mm(osfp.heatSinkFinPitch) * scale, y: h * 0.33, z: d / 2 + cageWall * 1.6 }, { x: cageWall * 0.5, y: h * 0.14, z: cageWall * 1.1 }, materials.rackEdge);
  }

  addCylinder(group, 'green link LED', { x: -w * 0.58, y: h * 0.58, z: d / 2 + cageWall }, ledRadius, cageWall, materials.ledGreen, { segments: 12 });
  addCylinder(group, 'amber activity LED', { x: -w * 0.45, y: h * 0.58, z: d / 2 + cageWall }, ledRadius * 0.88, cageWall, materials.ledAmber, { segments: 12 });

  maybeRegisterPort(portRegistry, {
    kind: 'osfp',
    x: position.x,
    y: position.y,
    z: position.z + d / 2 + cageWall * 0.25,
    scale
  });

  return group;
}

export function createE1SSlot({ materials, position = { x: 0, y: 0, z: 0 }, scale = 1 }) {
  const group = new THREE.Group();
  group.name = 'E1.S NVMe carrier slot';
  group.position.set(position.x, position.y, position.z);

  const e1s = PHYSICAL_DIMENSIONS_MM.e1s;
  const w = mm(e1s.frontSlotWidth) * scale;
  const h = mm(e1s.frontSlotHeight) * scale;
  const d = mm(3) * scale;
  const lipH = mm(e1s.carrierLipHeight) * scale;
  const latchH = mm(e1s.latchHeight) * scale;

  addRectangularPortFrame(group, 'E1.S open carrier guide', {
    width: w * 1.18,
    height: h,
    openingWidth: w * 0.76,
    openingHeight: h * 0.72,
    depth: d,
    material: materials.e1sLabel
  });
  addBox(group, 'E1.S recessed SSD slot tunnel', { x: 0, y: 0, z: -d * 0.35 }, { x: w * 0.68, y: h * 0.64, z: d * 0.65 }, materials.portVoid);
  addBox(group, 'E1.S carrier lip', { x: 0, y: h / 2 - lipH / 2, z: d / 2 + mm(0.6) }, { x: w * 1.18, y: lipH, z: mm(1.2) }, materials.e1sLabel);
  addBox(group, 'E1.S latch', { x: 0, y: -h / 2 + latchH / 2, z: d / 2 + mm(0.6) }, { x: w * 1.1, y: latchH, z: mm(1.2) }, materials.osfpMetal);
  addCylinder(group, 'E1.S retaining screw', { x: 0, y: h * 0.18, z: d / 2 + mm(0.9) }, mm(e1s.screwDiameter / 2) * scale, mm(0.6) * scale, materials.osfpMetal, { segments: 16 });

  return group;
}

export function createRJ45Port({ materials, position = { x: 0, y: 0, z: 0 }, scale = 1 }) {
  const group = new THREE.Group();
  group.name = 'RJ45 BMC management port';
  group.position.set(position.x, position.y, position.z);

  const rj45 = PHYSICAL_DIMENSIONS_MM.rj45;
  const w = mm(rj45.shellWidth) * scale;
  const h = mm(rj45.shellHeight) * scale;
  const d = mm(rj45.shellDepth) * scale;
  const jackW = mm(rj45.jackWidth) * scale;
  const jackH = mm(rj45.jackHeight) * scale;

  addRectangularPortFrame(group, 'RJ45 open metal shell', {
    width: w,
    height: h,
    openingWidth: jackW,
    openingHeight: jackH,
    depth: d,
    material: materials.osfpMetal
  });
  addBox(group, 'RJ45 recessed plug cavity', { x: 0, y: -h * 0.02, z: -d * 0.28 }, { x: jackW * 0.92, y: jackH * 0.86, z: d * 0.62 }, materials.portVoid);
  addBox(group, 'RJ45 clip notch', { x: 0, y: h * 0.32, z: d / 2 + mm(1.2) }, { x: jackW * 0.45, y: jackH * 0.2, z: mm(1.2) }, materials.trayDark);

  for (let i = 0; i < 8; i += 1) {
    const pinX = -w * 0.3 + i * (w * 0.085);
    addBox(group, 'RJ45 gold contact inside hollow jack', { x: pinX, y: -h * 0.23, z: d / 2 + mm(0.8) }, { x: w * 0.028, y: h * 0.12, z: mm(0.7) }, materials.gold, { castShadow: false });
  }

  addCylinder(group, 'RJ45 link LED', { x: -w * 0.62, y: h * 0.52, z: d / 2 + mm(1) }, mm(0.8) * scale, mm(0.5) * scale, materials.ledGreen, { segments: 12 });
  addCylinder(group, 'RJ45 activity LED', { x: w * 0.62, y: h * 0.52, z: d / 2 + mm(1) }, mm(0.8) * scale, mm(0.5) * scale, materials.ledAmber, { segments: 12 });

  return group;
}

export function createUSBCPort({ materials, position = { x: 0, y: 0, z: 0 }, scale = 1 }) {
  const group = new THREE.Group();
  group.name = 'USB-C service port';
  group.position.set(position.x, position.y, position.z);

  const usbC = PHYSICAL_DIMENSIONS_MM.usbC;
  addRoundedPortFrame(
    group,
    'USB-C open stainless rim with real void',
    { x: 0, y: 0, z: mm(2.4) * scale },
    mm(usbC.shellWidth) * scale,
    mm(usbC.shellHeight) * scale,
    mm(usbC.openingWidth) * scale,
    mm(usbC.openingHeight) * scale,
    mm(1.6) * scale,
    materials.osfpMetal
  );
  addBox(group, 'USB-C recessed port cavity', { x: 0, y: 0, z: mm(1.6) * scale }, { x: mm(usbC.openingWidth * 0.9) * scale, y: mm(usbC.openingHeight * 0.82) * scale, z: mm(1.2) * scale }, materials.portVoid);
  addBox(group, 'USB-C center tongue', { x: 0, y: 0, z: mm(3.05) * scale }, { x: mm(usbC.tongueWidth) * scale, y: mm(usbC.tongueHeight) * scale, z: mm(0.7) * scale }, materials.portVoid, { castShadow: false });

  return group;
}

export function createMiniSfpCage({ materials, position = { x: 0, y: 0, z: 0 }, scale = 1 }) {
  const group = new THREE.Group();
  group.name = 'small management SFP cage';
  group.position.set(position.x, position.y, position.z);

  const sfp = PHYSICAL_DIMENSIONS_MM.sfp;
  const w = mm(sfp.cageWidth) * scale;
  const h = mm(sfp.cageHeight) * scale;
  const d = mm(sfp.cageDepth) * scale;
  addRectangularPortFrame(group, 'SFP open cage', {
    width: w,
    height: h,
    openingWidth: w * 0.74,
    openingHeight: h * 0.6,
    depth: d,
    material: materials.osfpMetal
  });
  addBox(group, 'SFP recessed cage tunnel', { x: 0, y: 0, z: -d * 0.28 }, { x: w * 0.68, y: h * 0.54, z: d * 0.58 }, materials.portVoid);

  return group;
}

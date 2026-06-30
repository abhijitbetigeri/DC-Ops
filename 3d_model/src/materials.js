import * as THREE from './vendor/three.js';

export function createRackMaterials() {
  return {
    rackPaint: new THREE.MeshStandardMaterial({ color: 0x18202b, roughness: 0.72, metalness: 0.34 }),
    rackEdge: new THREE.MeshStandardMaterial({ color: 0x252f3c, roughness: 0.65, metalness: 0.42 }),
    innerBlack: new THREE.MeshStandardMaterial({ color: 0x05070a, roughness: 0.88, metalness: 0.18 }),
    champagneTrim: new THREE.MeshStandardMaterial({ color: 0xb9ad8a, roughness: 0.38, metalness: 0.66 }),
    blackAnodized: new THREE.MeshStandardMaterial({ color: 0x0b0f13, roughness: 0.52, metalness: 0.45 }),
    perforatedBlack: new THREE.MeshStandardMaterial({ color: 0x07090b, roughness: 0.78, metalness: 0.22 }),
    trayFace: new THREE.MeshStandardMaterial({ color: 0xc5c9c0, roughness: 0.42, metalness: 0.62 }),
    trayDark: new THREE.MeshStandardMaterial({ color: 0x171c1f, roughness: 0.7, metalness: 0.38 }),
    osfpMetal: new THREE.MeshStandardMaterial({ color: 0xd4d5ce, roughness: 0.28, metalness: 0.86 }),
    portVoid: new THREE.MeshStandardMaterial({ color: 0x020305, roughness: 0.95, metalness: 0.05 }),
    gold: new THREE.MeshStandardMaterial({ color: 0xf2bf53, roughness: 0.28, metalness: 0.7 }),
    copper: new THREE.MeshStandardMaterial({ color: 0xaa6437, roughness: 0.38, metalness: 0.64 }),
    pcb: new THREE.MeshStandardMaterial({ color: 0x244d3c, roughness: 0.8, metalness: 0.1 }),
    cable: new THREE.MeshStandardMaterial({ color: 0x050506, roughness: 0.52, metalness: 0.2 }),
    cableAlt: new THREE.MeshStandardMaterial({ color: 0x15171a, roughness: 0.48, metalness: 0.22 }),
    fiberCable: new THREE.MeshStandardMaterial({ color: 0xaeb3ad, roughness: 0.58, metalness: 0.03 }),
    dacCable: new THREE.MeshStandardMaterial({ color: 0x09090a, roughness: 0.54, metalness: 0.12 }),
    strainRelief: new THREE.MeshStandardMaterial({ color: 0x111315, roughness: 0.72, metalness: 0.08 }),
    ledGreen: new THREE.MeshStandardMaterial({ color: 0x7cff9b, emissive: 0x2dd85b, emissiveIntensity: 1.8 }),
    ledAmber: new THREE.MeshStandardMaterial({ color: 0xffc55c, emissive: 0xe8931f, emissiveIntensity: 1.4 }),
    ledBlue: new THREE.MeshStandardMaterial({ color: 0x7cc9ff, emissive: 0x2787ff, emissiveIntensity: 1.2 }),
    usbBlue: new THREE.MeshStandardMaterial({ color: 0x2678b8, roughness: 0.45, metalness: 0.15 }),
    e1sLabel: new THREE.MeshStandardMaterial({ color: 0x2b3034, roughness: 0.72, metalness: 0.24 }),
    floor: new THREE.MeshStandardMaterial({ color: 0x0d1117, roughness: 0.82, metalness: 0.08 })
  };
}

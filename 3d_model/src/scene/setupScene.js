import * as THREE from '../vendor/three.js';
import { addBox } from '../geometry/primitives.js';

export function setupScene({ materials }) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07090d);
  scene.fog = new THREE.Fog(0x07090d, 20, 54);

  addBox(
    scene,
    'matte data center floor',
    { x: 0, y: -5.05, z: 8 },
    { x: 30, y: 0.08, z: 42 },
    materials.floor,
    { castShadow: false, receiveShadow: true }
  );

  const keyLight = new THREE.DirectionalLight(0xf5f1df, 2.2);
  keyLight.position.set(2.8, 5.4, 4.5);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.width = 2048;
  keyLight.shadow.mapSize.height = 2048;
  keyLight.shadow.camera.near = 0.5;
  keyLight.shadow.camera.far = 55;
  keyLight.shadow.camera.left = -28;
  keyLight.shadow.camera.right = 28;
  keyLight.shadow.camera.top = 18;
  keyLight.shadow.camera.bottom = -7;

  const fillLight = new THREE.HemisphereLight(0xa8c8ff, 0x111318, 1.35);
  const rimLight = new THREE.PointLight(0x6ec2ff, 16, 9);
  rimLight.position.set(-3.6, 2.4, 3);

  scene.add(keyLight, fillLight, rimLight);
  return scene;
}

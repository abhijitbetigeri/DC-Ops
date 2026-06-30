import * as THREE from './vendor/three.js';
import { createRackMaterials } from './materials.js';
import { createNVL72Rack } from './model/createNVL72Rack.js';
import { setupScene } from './scene/setupScene.js';

const canvas = document.getElementById('rackCanvas');
const resetCameraButton = document.getElementById('resetCamera');
const toggleCablesButton = document.getElementById('toggleCables');
const toggleRotationButton = document.getElementById('toggleRotation');

const materials = createRackMaterials();
const scene = setupScene({ materials });
const rackModel = createNVL72Rack({ materials });
const aisleGroup = new THREE.Group();
aisleGroup.name = 'copy-paste datacenter aisle';
scene.add(aisleGroup);

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
camera.rotation.order = 'YXZ';

let cameraYaw = 0;
let cameraPitch = 0;
let autoRotate = false;
let cablesVisible = true;
const rackInstances = [];
const movementKeys = new Set();
const movementSpeed = 5.2;
const lookSensitivity = 0.003;
const zoomSpeed = 0.004;
const pointerState = {
  active: false,
  x: 0,
  y: 0
};
const aisle = {
  rowCenterX: 8.4,
  bayPitch: rackModel.layout.rack.width + 0.28,
  racksPerSide: 5
};

function setRackAisleTransform(root, slotIndex) {
  const bay = Math.floor(slotIndex / 2);
  const isLeftRow = slotIndex % 2 === 0;
  root.position.set(isLeftRow ? -aisle.rowCenterX : aisle.rowCenterX, 0, bay * aisle.bayPitch);
  root.rotation.set(0, isLeftRow ? Math.PI / 2 : -Math.PI / 2, 0);
}

function registerRackInstance(root) {
  const cables = root.getObjectByName('front patch cable set');
  if (cables) {
    cables.visible = cablesVisible;
  }
  rackInstances.push({ root, cables });
}

function buildDatacenterAisle() {
  for (let slotIndex = 0; slotIndex < aisle.racksPerSide * 2; slotIndex += 1) {
    const rackRoot = slotIndex === 0 ? rackModel.root : rackModel.root.clone(true);
    setRackAisleTransform(rackRoot, slotIndex);
    aisleGroup.add(rackRoot);
    registerRackInstance(rackRoot);
  }
}

function resetCamera() {
  const aisleMidZ = aisle.bayPitch * (aisle.racksPerSide - 1) * 0.5;
  camera.position.set(0, 0, aisleMidZ);
  camera.lookAt(0, 0, aisleMidZ + aisle.bayPitch);

  const cameraEuler = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
  cameraPitch = cameraEuler.x;
  cameraYaw = cameraEuler.y;
  applyCameraLook();
}

function resizeRenderer() {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

resetCameraButton.addEventListener('click', resetCamera);

toggleCablesButton.addEventListener('click', () => {
  cablesVisible = !cablesVisible;
  rackInstances.forEach((instance) => {
    if (instance.cables) {
      instance.cables.visible = cablesVisible;
    }
  });
  toggleCablesButton.textContent = cablesVisible ? 'Hide cables' : 'Show cables';
});

toggleRotationButton.addEventListener('click', () => {
  autoRotate = !autoRotate;
  toggleRotationButton.textContent = autoRotate ? 'Pause rotation' : 'Start rotation';
});

function applyCameraLook() {
  camera.rotation.set(cameraPitch, cameraYaw, 0);
}

function rotateCameraInPlace(deltaX, deltaY) {
  cameraYaw -= deltaX * lookSensitivity;
  cameraPitch -= deltaY * lookSensitivity;
  cameraPitch = THREE.MathUtils.clamp(cameraPitch, -Math.PI / 2 + 0.02, Math.PI / 2 - 0.02);
  applyCameraLook();
}

canvas.addEventListener('pointerdown', (event) => {
  pointerState.active = true;
  pointerState.x = event.clientX;
  pointerState.y = event.clientY;
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener('pointermove', (event) => {
  if (!pointerState.active) return;

  const deltaX = event.clientX - pointerState.x;
  const deltaY = event.clientY - pointerState.y;
  pointerState.x = event.clientX;
  pointerState.y = event.clientY;
  rotateCameraInPlace(deltaX, deltaY);
});

canvas.addEventListener('pointerup', (event) => {
  pointerState.active = false;
  if (canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
});

canvas.addEventListener('wheel', (event) => {
  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  camera.position.addScaledVector(forward, -event.deltaY * zoomSpeed);
  event.preventDefault();
}, { passive: false });

window.addEventListener('resize', resizeRenderer);
window.addEventListener('keydown', (event) => {
  const key = event.key.toLowerCase();
  if (['w', 'a', 's', 'd'].includes(key)) {
    movementKeys.add(key);
    event.preventDefault();
  }
});
window.addEventListener('keyup', (event) => {
  movementKeys.delete(event.key.toLowerCase());
});

buildDatacenterAisle();
resetCamera();
resizeRenderer();

const clock = new THREE.Clock();

function moveCameraWithKeyboard(delta) {
  if (movementKeys.size === 0) return;

  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  forward.y = 0;
  forward.normalize();

  const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
  const movement = new THREE.Vector3();

  if (movementKeys.has('w')) movement.add(forward);
  if (movementKeys.has('s')) movement.sub(forward);
  if (movementKeys.has('d')) movement.add(right);
  if (movementKeys.has('a')) movement.sub(right);

  if (movement.lengthSq() === 0) return;

  movement.normalize().multiplyScalar(movementSpeed * delta);
  camera.position.add(movement);
}

function animate() {
  const delta = clock.getDelta();
  if (autoRotate) {
    aisleGroup.rotation.y += delta * 0.08;
  }
  moveCameraWithKeyboard(delta);
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

animate();

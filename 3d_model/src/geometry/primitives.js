import * as THREE from '../vendor/three.js';

export function addBox(parent, name, position, size, material, options = {}) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(size.x, size.y, size.z), material);
  mesh.name = name;
  mesh.position.set(position.x, position.y, position.z);
  mesh.castShadow = options.castShadow ?? true;
  mesh.receiveShadow = options.receiveShadow ?? true;

  if (options.rotation) {
    mesh.rotation.set(options.rotation.x ?? 0, options.rotation.y ?? 0, options.rotation.z ?? 0);
  }

  parent.add(mesh);
  return mesh;
}

export function addCylinder(parent, name, position, radius, depth, material, options = {}) {
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, depth, options.segments ?? 24),
    material
  );
  mesh.name = name;
  mesh.position.set(position.x, position.y, position.z);
  mesh.rotation.x = options.rotationX ?? Math.PI / 2;
  mesh.castShadow = options.castShadow ?? true;
  mesh.receiveShadow = options.receiveShadow ?? true;
  parent.add(mesh);
  return mesh;
}

export function createRoundedRectShape(width, height, radius) {
  const x = -width / 2;
  const y = -height / 2;
  const shape = new THREE.Shape();
  shape.moveTo(x + radius, y);
  shape.lineTo(x + width - radius, y);
  shape.quadraticCurveTo(x + width, y, x + width, y + radius);
  shape.lineTo(x + width, y + height - radius);
  shape.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  shape.lineTo(x + radius, y + height);
  shape.quadraticCurveTo(x, y + height, x, y + height - radius);
  shape.lineTo(x, y + radius);
  shape.quadraticCurveTo(x, y, x + radius, y);
  return shape;
}

export function addFaceShape(parent, name, position, width, height, radius, material) {
  const geometry = new THREE.ShapeGeometry(createRoundedRectShape(width, height, radius));
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = name;
  mesh.position.set(position.x, position.y, position.z);
  parent.add(mesh);
  return mesh;
}

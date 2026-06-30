import * as THREE from '../vendor/three.js';
import { NVL72_LAYOUT, createNVL72TrayPlan } from '../config/nvl72Layout.js';
import { createRackShell } from '../components/rackShell.js';
import { createCableSet } from '../components/cables.js';
import { createComputeTray, createPowerShelf, createSwitchTray } from '../components/trays.js';

export function createNVL72Rack({ materials, layout = NVL72_LAYOUT } = {}) {
  const rackGroup = new THREE.Group();
  rackGroup.name = 'Vera Rubin NVL72 rack';

  const portRegistry = [];
  const shell = createRackShell({ materials, rack: layout.rack });
  const trays = new THREE.Group();
  trays.name = 'copyable tray stack';

  createNVL72TrayPlan(layout).forEach((item) => {
    if (item.type === 'computeTray') {
      trays.add(createComputeTray({
        materials,
        rack: layout.rack,
        y: item.y,
        height: item.height,
        index: item.index,
        portRegistry
      }));
    }

    if (item.type === 'switchTray') {
      trays.add(createSwitchTray({
        materials,
        rack: layout.rack,
        y: item.y,
        height: item.height,
        index: item.index,
        portRegistry
      }));
    }

    if (item.type === 'powerShelf') {
      trays.add(createPowerShelf({
        materials,
        rack: layout.rack,
        y: item.y,
        height: item.height,
        index: item.index
      }));
    }
  });

  const cables = createCableSet({ materials, rack: layout.rack, portRegistry });
  rackGroup.add(shell, trays, cables);

  return {
    root: rackGroup,
    shell,
    trays,
    cables,
    portRegistry,
    layout
  };
}

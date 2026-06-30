export const NVL72_LAYOUT = {
  rack: {
    width: 4.1,
    height: 9.7,
    depth: 2.3,
    frontZ: 1.18,
    innerWidth: 3.42,
    innerHeight: 8.95,
    railWidth: 0.13
  },
  trays: {
    computeCount: 18,
    switchCount: 9,
    powerShelfCount: 4,
    computeHeight: 0.285,
    switchHeight: 0.18,
    powerShelfHeight: 0.46,
    gap: 0.045
  },
  connectors: {
    computeTrayOsfpPerTray: 8,
    switchTrayOsfpPerTray: 32,
    registerCableEveryNthPort: 2
  }
};

export function createNVL72TrayPlan(layout = NVL72_LAYOUT) {
  const { rack, trays } = layout;
  const top = rack.innerHeight / 2 - 0.18;
  const bottom = -rack.innerHeight / 2 + 0.18;
  const shelfH = trays.powerShelfHeight;
  const computeH = trays.computeHeight;
  const switchH = trays.switchHeight;
  const gap = trays.gap;

  const plan = [];
  const bottomShelves = [
    bottom + shelfH * 0.5,
    bottom + shelfH * 1.5 + gap
  ];
  const topShelves = [
    top - shelfH * 1.5 - gap,
    top - shelfH * 0.5
  ];

  [...bottomShelves, ...topShelves].forEach((y, index) => {
    plan.push({ type: 'powerShelf', y, height: shelfH, index });
  });

  let computeIndex = 0;
  let y = bottomShelves[1] + shelfH * 0.5 + 0.24;
  for (let i = 0; i < trays.computeCount / 2; i += 1) {
    plan.push({ type: 'computeTray', y, height: computeH, index: computeIndex });
    y += computeH + gap;
    computeIndex += 1;
  }

  y += 0.2;
  for (let i = 0; i < trays.switchCount; i += 1) {
    plan.push({ type: 'switchTray', y, height: switchH, index: i });
    y += switchH + gap;
  }

  y += 0.2;
  for (let i = 0; i < trays.computeCount / 2; i += 1) {
    plan.push({ type: 'computeTray', y, height: computeH, index: computeIndex });
    y += computeH + gap;
    computeIndex += 1;
  }

  return plan;
}

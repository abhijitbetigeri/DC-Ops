const pptxgen = require("pptxgenjs");
const pres = new pptxgen();

// Theme colors
const BG_DARK = "0D1117";
const BG_CARD = "161B22";
const ACCENT = "00E676";
const ACCENT2 = "2196F3";
const ACCENT3 = "FF9800";
const TEXT_WHITE = "FFFFFF";
const TEXT_MUTED = "8B949E";
const TEXT_LIGHT = "C9D1D9";

pres.layout = "LAYOUT_WIDE";

// ============ SLIDE 1: TITLE ============
let slide1 = pres.addSlide();
slide1.background = { color: BG_DARK };
slide1.addText("DC-OPS", {
  x: 0.8, y: 1.0, w: 11, h: 1.5,
  fontSize: 72, fontFace: "Arial Black", color: ACCENT, bold: true,
});
slide1.addText("On-Device Data Center Operations Assistant", {
  x: 0.8, y: 2.5, w: 11, h: 0.8,
  fontSize: 28, fontFace: "Arial", color: TEXT_WHITE,
});
slide1.addText("Qualcomm x Meta ExecuTorch Hackathon — June 2026", {
  x: 0.8, y: 3.5, w: 11, h: 0.5,
  fontSize: 16, fontFace: "Arial", color: TEXT_MUTED, italic: true,
});
slide1.addText("Real-time AI-powered server rack inspection\nRunning entirely on Samsung Galaxy S25 Ultra — Snapdragon 8 Elite NPU", {
  x: 0.8, y: 4.5, w: 11, h: 0.9,
  fontSize: 14, fontFace: "Arial", color: TEXT_LIGHT, lineSpacingMultiple: 1.4,
});
slide1.addShape(pres.ShapeType.rect, {
  x: 0.8, y: 6.2, w: 2.5, h: 0.06, fill: { color: ACCENT },
});

// ============ SLIDE 2: PROBLEM ============
let slide2 = pres.addSlide();
slide2.background = { color: BG_DARK };
slide2.addText("THE PROBLEM", {
  x: 0.8, y: 0.4, w: 11, h: 0.7,
  fontSize: 36, fontFace: "Arial Black", color: ACCENT, bold: true,
});

const problems = [
  { icon: "🔒", title: "Air-Gapped Environments", desc: "Data centers prohibit cloud connectivity in server rooms — no API calls possible" },
  { icon: "🕐", title: "Slow Manual Inspection", desc: "Technicians walk aisles checking thousands of LEDs, ports, and cables by eye" },
  { icon: "🔐", title: "Sensitive Infrastructure Data", desc: "Serial numbers, rack topology, and server status are classified information" },
  { icon: "📶", title: "No Connectivity", desc: "Even connected DCs have dead zones between rack rows — WiFi is unreliable" },
];

problems.forEach((p, i) => {
  const y = 1.5 + i * 1.2;
  slide2.addShape(pres.ShapeType.roundRect, {
    x: 0.8, y: y, w: 11.4, h: 1.0,
    fill: { color: BG_CARD }, rectRadius: 0.1,
  });
  slide2.addText(p.icon, { x: 1.0, y: y + 0.1, w: 0.8, h: 0.8, fontSize: 28 });
  slide2.addText(p.title, {
    x: 1.9, y: y + 0.05, w: 9, h: 0.45,
    fontSize: 18, fontFace: "Arial", color: TEXT_WHITE, bold: true,
  });
  slide2.addText(p.desc, {
    x: 1.9, y: y + 0.5, w: 9, h: 0.4,
    fontSize: 13, fontFace: "Arial", color: TEXT_MUTED,
  });
});

// ============ SLIDE 3: SOLUTION ============
let slide3 = pres.addSlide();
slide3.background = { color: BG_DARK };
slide3.addText("THE SOLUTION", {
  x: 0.8, y: 0.4, w: 11, h: 0.7,
  fontSize: 36, fontFace: "Arial Black", color: ACCENT, bold: true,
});
slide3.addText("Point your phone at any server rack.\nAI identifies every component in real-time.\nAll processing happens on-device — zero cloud.", {
  x: 0.8, y: 1.3, w: 11, h: 1.0,
  fontSize: 20, fontFace: "Arial", color: TEXT_WHITE, lineSpacingMultiple: 1.5,
});

const features = [
  { title: "16 Component Classes", desc: "Compute trays, ports, LEDs, cables, fans, drives, PSUs, DPUs, and more", color: ACCENT },
  { title: "RAG Knowledge Overlay", desc: "Tap any detection → specs, troubleshooting, LED meanings, maintenance steps", color: ACCENT2 },
  { title: "CPU vs NPU Toggle", desc: "Live comparison: XNNPACK (CPU) ↔ QNN HTP (Snapdragon NPU) with FPS/latency", color: ACCENT3 },
];

features.forEach((f, i) => {
  const x = 0.8 + i * 4.0;
  slide3.addShape(pres.ShapeType.roundRect, {
    x: x, y: 3.0, w: 3.6, h: 2.5,
    fill: { color: BG_CARD }, rectRadius: 0.1,
  });
  slide3.addShape(pres.ShapeType.rect, {
    x: x, y: 3.0, w: 3.6, h: 0.06, fill: { color: f.color },
  });
  slide3.addText(f.title, {
    x: x + 0.2, y: 3.3, w: 3.2, h: 0.6,
    fontSize: 18, fontFace: "Arial", color: TEXT_WHITE, bold: true,
  });
  slide3.addText(f.desc, {
    x: x + 0.2, y: 4.0, w: 3.2, h: 1.2,
    fontSize: 13, fontFace: "Arial", color: TEXT_MUTED, lineSpacingMultiple: 1.3,
  });
});

// ============ SLIDE 4: WHY ON-DEVICE ============
let slide4 = pres.addSlide();
slide4.background = { color: BG_DARK };
slide4.addText("WHY ON-DEVICE?", {
  x: 0.8, y: 0.4, w: 11, h: 0.7,
  fontSize: 36, fontFace: "Arial Black", color: ACCENT, bold: true,
});

const reasons = [
  { stat: "0ms", label: "Cloud Latency", desc: "No round-trip — inference directly on Hexagon NPU" },
  { stat: "100%", label: "Offline", desc: "Works in air-gapped rooms with zero connectivity" },
  { stat: "0 bytes", label: "Data Leaked", desc: "Serial numbers and infra data never leave the device" },
  { stat: "4-5x", label: "Power Efficient", desc: "NPU vs CPU — longer battery life for extended inspections" },
];

reasons.forEach((r, i) => {
  const x = 0.8 + i * 3.0;
  slide4.addShape(pres.ShapeType.roundRect, {
    x: x, y: 1.5, w: 2.7, h: 3.5,
    fill: { color: BG_CARD }, rectRadius: 0.1,
  });
  slide4.addText(r.stat, {
    x: x, y: 1.8, w: 2.7, h: 1.0,
    fontSize: 48, fontFace: "Arial Black", color: ACCENT, bold: true, align: "center",
  });
  slide4.addText(r.label, {
    x: x, y: 2.8, w: 2.7, h: 0.5,
    fontSize: 16, fontFace: "Arial", color: TEXT_WHITE, bold: true, align: "center",
  });
  slide4.addText(r.desc, {
    x: x + 0.2, y: 3.4, w: 2.3, h: 1.2,
    fontSize: 12, fontFace: "Arial", color: TEXT_MUTED, align: "center", lineSpacingMultiple: 1.3,
  });
});

// ============ SLIDE 5: ARCHITECTURE ============
let slide5 = pres.addSlide();
slide5.background = { color: BG_DARK };
slide5.addText("ARCHITECTURE", {
  x: 0.8, y: 0.4, w: 11, h: 0.7,
  fontSize: 36, fontFace: "Arial Black", color: ACCENT, bold: true,
});

const archSteps = [
  { label: "Camera\n(CameraX 30fps)", color: ACCENT2, x: 0.5, y: 1.8 },
  { label: "RetinaNet\n(INT8 Quantized)", color: ACCENT, x: 3.2, y: 1.8 },
  { label: "QNN HTP\n(Snapdragon NPU)", color: ACCENT3, x: 5.9, y: 1.8 },
  { label: "16 DC Classes\n(Polygon Overlay)", color: "E040FB", x: 8.6, y: 1.8 },
];

archSteps.forEach((s, i) => {
  slide5.addShape(pres.ShapeType.roundRect, {
    x: s.x, y: s.y, w: 2.4, h: 1.2,
    fill: { color: BG_CARD }, rectRadius: 0.1,
    line: { color: s.color, width: 2 },
  });
  slide5.addText(s.label, {
    x: s.x, y: s.y + 0.1, w: 2.4, h: 1.0,
    fontSize: 13, fontFace: "Arial", color: TEXT_WHITE, align: "center", bold: true,
  });
  if (i < archSteps.length - 1) {
    slide5.addText("→", {
      x: s.x + 2.4, y: s.y + 0.2, w: 0.8, h: 0.8,
      fontSize: 28, color: TEXT_MUTED, align: "center",
    });
  }
});

// RAG pipeline below
slide5.addShape(pres.ShapeType.roundRect, {
  x: 3.2, y: 3.5, w: 6.3, h: 1.0,
  fill: { color: BG_CARD }, rectRadius: 0.1,
  line: { color: ACCENT2, width: 1 },
});
slide5.addText("RAG Knowledge Overlay: CLIP ViT-B-32 → FAISS (160KB) → Specs, Troubleshooting, LED Meanings", {
  x: 3.4, y: 3.55, w: 5.9, h: 0.9,
  fontSize: 12, fontFace: "Arial", color: TEXT_LIGHT, align: "center",
});

// 3D model callout
slide5.addShape(pres.ShapeType.roundRect, {
  x: 0.5, y: 3.5, w: 2.4, h: 1.0,
  fill: { color: BG_CARD }, rectRadius: 0.1,
  line: { color: "FF5722", width: 1 },
});
slide5.addText("3D NVL72 Rack\n(Three.js Interactive)", {
  x: 0.5, y: 3.55, w: 2.4, h: 0.9,
  fontSize: 12, fontFace: "Arial", color: TEXT_LIGHT, align: "center", bold: true,
});

slide5.addText("Android App → ExecuTorch Runtime → QAIRT SDK → Snapdragon NPU", {
  x: 0.8, y: 5.0, w: 11, h: 0.5,
  fontSize: 14, fontFace: "Consolas", color: TEXT_MUTED, align: "center",
});

// ============ SLIDE 6: TECH STACK ============
let slide6 = pres.addSlide();
slide6.background = { color: BG_DARK };
slide6.addText("TECH STACK", {
  x: 0.8, y: 0.4, w: 11, h: 0.7,
  fontSize: 36, fontFace: "Arial Black", color: ACCENT, bold: true,
});

const stack = [
  ["Detection (NPU)", "RetinaNet-ResNet50-FPN", "QNN HTP, INT8, 36MB"],
  ["Detection (CPU)", "YOLOv8n-seg v3", "XNNPACK, FP32, 13MB"],
  ["RAG Retrieval", "CLIP ViT-B-32 + FAISS", "160KB index, on-device"],
  ["Runtime", "ExecuTorch 1.4", "QNN backend for SM8750"],
  ["App", "Kotlin + CameraX", "ExecuTorch AAR (3.4MB)"],
  ["Device", "Galaxy S25 Ultra", "Snapdragon 8 Elite, Hexagon v79"],
  ["3D Visualization", "Three.js NVL72 Model", "Interactive rack explorer"],
];

stack.forEach((row, i) => {
  const y = 1.3 + i * 0.65;
  const bgColor = i % 2 === 0 ? BG_CARD : BG_DARK;
  slide6.addShape(pres.ShapeType.rect, {
    x: 0.8, y: y, w: 11.4, h: 0.6, fill: { color: bgColor },
  });
  slide6.addText(row[0], {
    x: 0.9, y: y, w: 3.0, h: 0.6,
    fontSize: 14, fontFace: "Arial", color: ACCENT, bold: true, valign: "middle",
  });
  slide6.addText(row[1], {
    x: 4.0, y: y, w: 4.0, h: 0.6,
    fontSize: 14, fontFace: "Arial", color: TEXT_WHITE, valign: "middle",
  });
  slide6.addText(row[2], {
    x: 8.2, y: y, w: 3.8, h: 0.6,
    fontSize: 12, fontFace: "Arial", color: TEXT_MUTED, valign: "middle",
  });
});

// ============ SLIDE 7: TRAINING PIPELINE ============
let slide7 = pres.addSlide();
slide7.background = { color: BG_DARK };
slide7.addText("TRAINING PIPELINE", {
  x: 0.8, y: 0.4, w: 11, h: 0.7,
  fontSize: 36, fontFace: "Arial Black", color: ACCENT, bold: true,
});

slide7.addText("2,036 human-labeled images from 4 Roboflow datasets (CC BY 4.0)", {
  x: 0.8, y: 1.2, w: 11, h: 0.5,
  fontSize: 16, fontFace: "Arial", color: TEXT_WHITE,
});

const datasets = [
  { name: "Server Vision", imgs: "1,293", classes: "30 classes (CPU, DIMM, fans, PSUs, drives)", pct: 64 },
  { name: "PC Ports", imgs: "327", classes: "4 classes (HDMI, RJ45, USB-A, USB-C)", pct: 16 },
  { name: "Ports (ym)", imgs: "138", classes: "7 classes (DP, ethernet, power, USB)", pct: 7 },
  { name: "Server Detection", imgs: "54", classes: "9 classes (SFP, XFP, patch panels)", pct: 3 },
];

datasets.forEach((d, i) => {
  const y = 2.0 + i * 0.9;
  slide7.addText(d.name, {
    x: 0.8, y: y, w: 2.5, h: 0.5,
    fontSize: 14, fontFace: "Arial", color: TEXT_WHITE, bold: true,
  });
  slide7.addText(d.imgs + " imgs", {
    x: 3.3, y: y, w: 1.5, h: 0.5,
    fontSize: 14, fontFace: "Arial", color: ACCENT,
  });
  slide7.addText(d.classes, {
    x: 4.8, y: y, w: 5.0, h: 0.5,
    fontSize: 12, fontFace: "Arial", color: TEXT_MUTED,
  });
  // progress bar
  slide7.addShape(pres.ShapeType.rect, {
    x: 10.0, y: y + 0.15, w: 2.0, h: 0.2, fill: { color: BG_CARD },
  });
  slide7.addShape(pres.ShapeType.rect, {
    x: 10.0, y: y + 0.15, w: 2.0 * d.pct / 100, h: 0.2, fill: { color: ACCENT },
  });
});

slide7.addText("50+ source classes → mapped to 16 DC-Ops classes\nAugmentations: Moiré patterns, screen glare, brightness variation, color tint, scanlines", {
  x: 0.8, y: 5.0, w: 11, h: 0.8,
  fontSize: 13, fontFace: "Arial", color: TEXT_LIGHT, lineSpacingMultiple: 1.4,
});

// ============ SLIDE 8: MODEL PERFORMANCE ============
let slide8 = pres.addSlide();
slide8.background = { color: BG_DARK };
slide8.addText("MODEL PERFORMANCE", {
  x: 0.8, y: 0.4, w: 11, h: 0.7,
  fontSize: 36, fontFace: "Arial Black", color: ACCENT, bold: true,
});

// Big stats
const stats = [
  { val: "0.749", label: "mAP50 (box)", sub: "YOLOv8n-seg v3" },
  { val: "0.85", label: "Final Loss", sub: "RetinaNet (moiré aug)" },
  { val: "6.4ms", label: "Inference", sub: "Per frame on T4 GPU" },
];

stats.forEach((s, i) => {
  const x = 0.8 + i * 4.0;
  slide8.addShape(pres.ShapeType.roundRect, {
    x: x, y: 1.3, w: 3.6, h: 1.8,
    fill: { color: BG_CARD }, rectRadius: 0.1,
  });
  slide8.addText(s.val, {
    x: x, y: 1.4, w: 3.6, h: 0.9,
    fontSize: 44, fontFace: "Arial Black", color: ACCENT, bold: true, align: "center",
  });
  slide8.addText(s.label, {
    x: x, y: 2.2, w: 3.6, h: 0.4,
    fontSize: 16, fontFace: "Arial", color: TEXT_WHITE, align: "center", bold: true,
  });
  slide8.addText(s.sub, {
    x: x, y: 2.6, w: 3.6, h: 0.35,
    fontSize: 12, fontFace: "Arial", color: TEXT_MUTED, align: "center",
  });
});

// Per-class table
const classes = [
  ["compute tray", "0.924"], ["network port", "0.912"], ["power shelf", "0.832"],
  ["LED indicator", "0.808"], ["server rack", "0.683"], ["drive bay", "0.912"],
  ["DPU", "0.808"], ["cooling manifold", "0.833"],
];

classes.forEach((c, i) => {
  const col = i < 4 ? 0 : 1;
  const row = i % 4;
  const x = 0.8 + col * 6.0;
  const y = 3.5 + row * 0.55;
  slide8.addText(c[0], {
    x: x, y: y, w: 3.0, h: 0.5,
    fontSize: 13, fontFace: "Arial", color: TEXT_WHITE,
  });
  slide8.addText(c[1], {
    x: x + 3.0, y: y, w: 1.5, h: 0.5,
    fontSize: 13, fontFace: "Consolas", color: ACCENT,
  });
});

// ============ SLIDE 9: DEMO FLOW ============
let slide9 = pres.addSlide();
slide9.background = { color: BG_DARK };
slide9.addText("LIVE DEMO", {
  x: 0.8, y: 0.4, w: 11, h: 0.7,
  fontSize: 36, fontFace: "Arial Black", color: ACCENT, bold: true,
});

const demoSteps = [
  { time: "0:00", action: "Point phone at server rack / mini PC", detail: "Real-time colored polygon overlays appear" },
  { time: "1:00", action: "Tap detected component", detail: "RAG info panel: specs, troubleshooting, LED meanings" },
  { time: "2:00", action: "Toggle CPU ↔ NPU", detail: "Live FPS/latency comparison on screen" },
  { time: "3:00", action: "Enable airplane mode", detail: "Everything still works — fully offline" },
  { time: "4:00", action: "3D rack explorer", detail: "Interactive NVL72 model with component details" },
];

demoSteps.forEach((d, i) => {
  const y = 1.3 + i * 0.95;
  slide9.addShape(pres.ShapeType.roundRect, {
    x: 0.8, y: y, w: 1.2, h: 0.75,
    fill: { color: ACCENT }, rectRadius: 0.1,
  });
  slide9.addText(d.time, {
    x: 0.8, y: y, w: 1.2, h: 0.75,
    fontSize: 16, fontFace: "Consolas", color: BG_DARK, align: "center", bold: true, valign: "middle",
  });
  slide9.addText(d.action, {
    x: 2.2, y: y, w: 5.0, h: 0.45,
    fontSize: 16, fontFace: "Arial", color: TEXT_WHITE, bold: true,
  });
  slide9.addText(d.detail, {
    x: 2.2, y: y + 0.4, w: 9.0, h: 0.35,
    fontSize: 12, fontFace: "Arial", color: TEXT_MUTED,
  });
});

// ============ SLIDE 10: RAG KNOWLEDGE OVERLAY ============
let slide10 = pres.addSlide();
slide10.background = { color: BG_DARK };
slide10.addText("RAG KNOWLEDGE OVERLAY", {
  x: 0.8, y: 0.4, w: 11, h: 0.7,
  fontSize: 36, fontFace: "Arial Black", color: ACCENT, bold: true,
});

slide10.addText("Detect → Embed → Retrieve → Display", {
  x: 0.8, y: 1.2, w: 11, h: 0.5,
  fontSize: 18, fontFace: "Arial", color: TEXT_WHITE,
});

// Example info panel
slide10.addShape(pres.ShapeType.roundRect, {
  x: 1.5, y: 2.0, w: 5.5, h: 3.5,
  fill: { color: BG_CARD }, rectRadius: 0.15,
  line: { color: ACCENT, width: 2 },
});
slide10.addText("COMPUTE TRAY                           92%", {
  x: 1.7, y: 2.1, w: 5.1, h: 0.5,
  fontSize: 16, fontFace: "Consolas", color: ACCENT, bold: true,
});
slide10.addText(
  "⚙️  2x Grace CPU, 4x B200 GPU, 192GB HBM3e\n\n" +
  "🟢  Green LED = healthy and operational\n" +
  "🟡  Amber LED = check BMC log, verify coolant\n" +
  "🔴  Red LED = critical hardware failure\n\n" +
  "🔧  Maintenance: check cold plate contact,\n     verify NVLink connector seating",
  {
    x: 1.7, y: 2.7, w: 5.1, h: 2.6,
    fontSize: 13, fontFace: "Consolas", color: TEXT_LIGHT, lineSpacingMultiple: 1.3,
  }
);

// Pipeline on right
const ragSteps = [
  "80 knowledge chunks",
  "CLIP ViT-B-32 embeddings",
  "FAISS vector index (160KB)",
  "NVL72-specific documentation",
  "All on-device, zero cloud",
];
ragSteps.forEach((s, i) => {
  slide10.addText("▸ " + s, {
    x: 7.5, y: 2.2 + i * 0.55, w: 4.5, h: 0.5,
    fontSize: 14, fontFace: "Arial", color: TEXT_LIGHT,
  });
});

// ============ SLIDE 11: CPU vs NPU ============
let slide11 = pres.addSlide();
slide11.background = { color: BG_DARK };
slide11.addText("CPU vs NPU COMPARISON", {
  x: 0.8, y: 0.4, w: 11, h: 0.7,
  fontSize: 36, fontFace: "Arial Black", color: ACCENT, bold: true,
});

// Two columns
const comparisons = [
  ["Backend", "XNNPACK (CPU)", "QNN HTP (NPU)"],
  ["Model", "YOLOv8n-seg", "RetinaNet"],
  ["Size", "13 MB", "36 MB"],
  ["Quantization", "FP32", "INT8"],
  ["Est. Latency", "~15-30ms", "~3-5ms"],
  ["Est. FPS", "~30-60", "~100+"],
  ["Power", "Higher drain", "4-5x efficient"],
  ["Hardware", "ARM Kryo cores", "Hexagon HTP v79"],
];

comparisons.forEach((row, i) => {
  const y = 1.3 + i * 0.58;
  const bg = i === 0 ? "1E2761" : (i % 2 === 0 ? BG_CARD : BG_DARK);
  const tc = i === 0 ? ACCENT : TEXT_WHITE;
  const bold = i === 0;
  slide11.addShape(pres.ShapeType.rect, {
    x: 0.8, y: y, w: 11.4, h: 0.55, fill: { color: bg },
  });
  slide11.addText(row[0], {
    x: 0.9, y: y, w: 3.5, h: 0.55,
    fontSize: 14, fontFace: "Arial", color: tc, bold: bold, valign: "middle",
  });
  slide11.addText(row[1], {
    x: 4.5, y: y, w: 3.5, h: 0.55,
    fontSize: 14, fontFace: "Arial", color: i === 0 ? tc : TEXT_MUTED, bold: bold, valign: "middle",
  });
  slide11.addText(row[2], {
    x: 8.0, y: y, w: 4.0, h: 0.55,
    fontSize: 14, fontFace: "Arial", color: i === 0 ? tc : ACCENT, bold: true, valign: "middle",
  });
});

// ============ SLIDE 12: WHAT'S NEXT ============
let slide12 = pres.addSlide();
slide12.background = { color: BG_DARK };
slide12.addText("WHAT'S NEXT", {
  x: 0.8, y: 0.4, w: 11, h: 0.7,
  fontSize: 36, fontFace: "Arial Black", color: ACCENT, bold: true,
});

const futures = [
  { title: "Fine-grained Port Classification", desc: "Split 'network port' into HDMI, USB-A, USB-C, RJ45, SFP individually" },
  { title: "LED State Detection", desc: "Classify LED colors (green/amber/red) and blink patterns for health scoring" },
  { title: "On-Device LLM Integration", desc: "LLaMA 3.2 1B via ExecuTorch for natural language troubleshooting Q&A" },
  { title: "Audit Log + Reporting", desc: "SQLite database of all inspections, exportable via secure channel" },
  { title: "Multi-Rack Mapping", desc: "Combine depth estimation + detection for spatial rack layout understanding" },
];

futures.forEach((f, i) => {
  const y = 1.3 + i * 0.95;
  slide12.addShape(pres.ShapeType.rect, {
    x: 0.8, y: y + 0.05, w: 0.06, h: 0.7, fill: { color: ACCENT },
  });
  slide12.addText(f.title, {
    x: 1.2, y: y, w: 10, h: 0.45,
    fontSize: 18, fontFace: "Arial", color: TEXT_WHITE, bold: true,
  });
  slide12.addText(f.desc, {
    x: 1.2, y: y + 0.45, w: 10, h: 0.4,
    fontSize: 13, fontFace: "Arial", color: TEXT_MUTED,
  });
});

// ============ SLIDE 13: TEAM + LINKS ============
let slide13 = pres.addSlide();
slide13.background = { color: BG_DARK };
slide13.addText("DC-OPS", {
  x: 0.8, y: 0.4, w: 11, h: 0.8,
  fontSize: 48, fontFace: "Arial Black", color: ACCENT, bold: true,
});
slide13.addText("On-Device Data Center Operations Assistant", {
  x: 0.8, y: 1.2, w: 11, h: 0.5,
  fontSize: 20, fontFace: "Arial", color: TEXT_WHITE,
});

slide13.addShape(pres.ShapeType.rect, {
  x: 0.8, y: 2.0, w: 11.4, h: 0.02, fill: { color: "30363D" },
});

// Links
const links = [
  ["GitHub", "github.com/abhijitbetigeri/DC-Ops"],
  ["HuggingFace", "huggingface.co/datasets/abhijitbetigeri/dc-ops-dataset"],
  ["Models", "RetinaNet QNN HTP (.pte) + YOLOv8n-seg (.pte) + RAG index"],
];

links.forEach((l, i) => {
  const y = 2.3 + i * 0.6;
  slide13.addText(l[0], {
    x: 0.8, y: y, w: 2.5, h: 0.5,
    fontSize: 16, fontFace: "Arial", color: ACCENT, bold: true,
  });
  slide13.addText(l[1], {
    x: 3.3, y: y, w: 8.5, h: 0.5,
    fontSize: 14, fontFace: "Consolas", color: TEXT_LIGHT,
  });
});

slide13.addText("Built with ExecuTorch + Qualcomm QNN HTP + Snapdragon 8 Elite", {
  x: 0.8, y: 4.5, w: 11, h: 0.5,
  fontSize: 14, fontFace: "Arial", color: TEXT_MUTED, italic: true,
});

slide13.addText("Thank you!", {
  x: 0.8, y: 5.5, w: 11, h: 0.7,
  fontSize: 32, fontFace: "Arial Black", color: ACCENT,
});

// Save
const outPath = "/Users/abhijitbetigeri/projects/DC-Ops/presentation/dc_ops_pitch.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("Saved to " + outPath);
});

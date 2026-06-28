# DC-Ops — Architecture & Team Working Agreement

> **Read this first.** This is the contract that lets 4 people clone the repo and build in
> parallel without colliding or producing pieces that don't fit together. If you change
> anything in the **Contracts** section, tell the whole team — those are frozen interfaces
> that other people are coding against.

**Hackathon:** Qualcomm × Meta ExecuTorch — June 27–28, 2026
**Target device:** Samsung Galaxy S25 Ultra · Snapdragon 8 Elite (SM8750-AC) · Hexagon v79

---

## 1. The one idea behind this doc

Parallel work fails for two reasons: **people edit the same files**, and **components don't fit
at integration time**. We prevent both with three rules:

1. **Boundaries** — every person owns distinct directories/packages. You never edit someone
   else's package.
2. **Contracts** — the data that crosses a boundary has a frozen shape (Section 5). You code
   against the *contract*, not against the other person's implementation.
3. **Stub-first** — every component ships a fake/stub from day one so everyone downstream is
   unblocked immediately. Real implementations swap in behind the same contract.

The Android branch already proves this works: `ModelManager` returns **fake detections** today,
so the camera + overlay pipeline is fully demo-able *before the YOLO model exists*.

---

## 2. System overview

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                      android-app  (the host)                         │
 │                                                                      │
 │  CameraX ──frame──▶ ModelManager ──List<DetectionResult>──▶ Overlay  │
 │  [Stream A]         [Stream A seam]      (contract §5.3)     [Stream C]│
 │                          │                                           │
 │                          │ loads  yolov8n_dcops_int8.pte  (§5.4)     │
 │                          ▼                                           │
 │                    ExecuTorch AAR ──▶ QNN ──▶ Snapdragon NPU         │
 │                          │                                           │
 │                     Finding (§5.5)                                   │
 │                          ▼                                           │
 │                  FindingsRepository ──▶ SQLite     ──▶ Audit-log UI  │
 │                     [Stream D]                          [Stream C]   │
 └─────────────────────────────────────────────────────────────────────┘
                              ▲
                              │  produces the .pte artifact
            ┌─────────────────┴──────────────────┐
            │  models/  + data/                  │
            │  YOLO training/export  [Stream B]  │
            │  dataset + labels      [Stream D]  │
            └────────────────────────────────────┘
```

---

## 3. Workstreams & ownership

| # | Stream | Owner | Owns (you edit ONLY these) | Branch |
|---|--------|-------|----------------------------|--------|
| **A** | **Android host & integration** | Android dev | `android-app/.../MainActivity.kt`, `camera/`, `inference/` (incl. the `ModelManager` seam), ExecuTorch AAR wiring, Gradle | `name/android-core` |
| **B** | **ML / YOLO model** | Model engineer | `models/` (train + export), the class taxonomy & output spec, the `.pte` artifact | `name/yolo-model` |
| **C** | **App UI/UX & screens** | UI dev | `android-app/.../ui/`, `android-app/.../overlay/`, `res/layout/`, `res/values/`, themes/icons | `name/app-ui` |
| **D** | **Data · Persistence · Demo glue** | "Everything else" | `data/` (images+labels), `android-app/.../data/` (SQLite), demo script, CI, merge integration | `name/data-persist` |

**Why D is one person, not three jobs:** dataset/labeling feeds B, SQLite is the other half of
the product, and *someone* must own merges + the demo or it falls through the cracks. If you grow
to 5 people, split "Data pipeline" off first.

### Shared files (the collision zone)

These files are touched by A, C, and D. To avoid merge wars, **Stream A (Android dev) is the
integrator** for them — others request changes via PR or Slack, A merges:

- `MainActivity.kt` — the wiring hub
- `app/build.gradle.kts` — dependencies
- `AndroidManifest.xml`
- `res/values/strings.xml`, `colors.xml`, `themes.xml`

**Design rule that minimizes this:** keep `MainActivity` *thin*. Each feature exposes **one entry
class** (e.g. `AuditLogScreen`, `FindingsRepository`) that A wires in with a single line. You build
behind your class; A plugs it in.

---

## 4. Repository layout

```
DC-Ops/
├── ARCHITECTURE.md              ← this file (lives on main; everyone reads it)
├── README.md
├── android-app/                 ← the Android host
│   └── app/src/main/
│       ├── java/com/dcops/ar/
│       │   ├── MainActivity.kt         [A] wiring hub — thin
│       │   ├── camera/                 [A] CameraX
│       │   ├── inference/              [A] ModelManager seam + DetectionResult (§5.3)
│       │   ├── overlay/                [C] AR polygon overlay
│       │   ├── ui/                     [C] screens, audit-log view, settings   ← NEW
│       │   └── data/                   [D] SQLite, FindingsRepository           ← NEW
│       ├── assets/models/              [B→A] the .pte goes here (§5.4)          ← NEW
│       └── res/                        [C] layouts, values, drawables
├── models/                      ← [B]
│   ├── train/                   training scripts, dataset yaml, hyperparams    ← NEW
│   ├── export/                  PyTorch → .pte (export_yolo.py, export_ocr.py)
│   └── classes.yaml             FROZEN class taxonomy (§5.1) — single source   ← NEW
├── data/                        ← [D] sample_images/, labels/, dataset splits  ← NEW
└── scripts/                     setup_env.sh, build helpers
```

> **Naming note:** the Android folder is `android-app/` (not `android/app/` as the README/CLAUDE.md
> say). Pick one and fix the docs — this is an open item in Section 9.

---

## 5. Contracts (FROZEN — change only by team agreement)

This is the core of the document. Each contract is the *only* thing two streams need to agree on.

### 5.1 Class taxonomy — FROZEN

The single source of truth for `classId ↔ label`. Lives in `models/classes.yaml`; the Android side
mirrors it as an enum. **B trains to these exact IDs; A and C render them.**

```yaml
# models/classes.yaml
0: led_green
1: led_amber
2: led_red
3: led_off
4: cable
5: label        # nameplate / serial-number region (feeds OCR stretch goal)
```

⚠️ **Known mismatch to fix:** the current stub `ModelManager` uses `0=LED-green, 1=label, 2=cable`.
Align the stub to the table above as the first task on Stream A.

### 5.2 Frame format (Camera → Model) — A↔B boundary

| Property | Value |
|----------|-------|
| Source | CameraX `ImageAnalysis`, `STRATEGY_KEEP_ONLY_LATEST` |
| Camera resolution | 1280×720, back camera |
| Pixel format | `YUV_420_888` (`ImageProxy`) |
| Model input | NCHW `float32`, **640×640**, RGB, normalized `0–1` (letterboxed, aspect-preserved) |

A owns the YUV→RGB→resize→normalize conversion. B documents the exact normalization its training
used (mean/std or plain `/255`) so A matches it.

### 5.3 Detection result (Model → App) — ALREADY IN CODE ✅

`com.dcops.ar.inference.DetectionResult` — **this already exists and is the canonical contract.**

```kotlin
data class DetectionResult(
    val label: String,           // human-readable, from classes.yaml
    val score: Float,            // confidence 0..1
    val polygon: List<PointF>,   // NORMALIZED 0..1 image coords; 4 pts = box, >4 = mask
    val classId: Int = -1        // index from classes.yaml (§5.1)
)
```

- Coordinates are **normalized 0–1** so the overlay scales to any view size. Never pass pixels.
- B+A produce a `List<DetectionResult>`; C consumes it in the overlay. Neither side needs to know
  the other's internals.

### 5.4 Model artifact (B → A)

| Property | Value |
|----------|-------|
| Filename | `yolov8n_dcops_int8.pte` |
| Location in app | `android-app/app/src/main/assets/models/` |
| Quantization | INT8 (post-training or QAT), QNN backend, `soc_model=SM8750` |
| Output spec | B documents raw output tensor shape + how to decode → `DetectionResult` |

**`.pte` files are gitignored** (binary, large). Share the artifact via **GitHub Release asset** or
a shared Drive link, *not* git. A pulls the latest into `assets/models/` before building.
Until B delivers, A keeps the stub.

### 5.5 Finding (App → Persistence) — D defines

The audit-log record written when a technician captures/confirms a detection.

```kotlin
data class Finding(
    val id: Long = 0,
    val timestampMs: Long,
    val classId: Int,            // §5.1
    val label: String,
    val score: Float,
    val bbox: List<PointF>,      // normalized, from DetectionResult
    val serial: String? = null,  // OCR result (stretch, §5.7)
    val imageRef: String? = null // optional saved-frame path
)
```

SQLite schema (D owns the DDL; this is the shape A/C rely on):

```sql
CREATE TABLE findings (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp  INTEGER NOT NULL,
  class_id   INTEGER NOT NULL,
  label      TEXT    NOT NULL,
  score      REAL    NOT NULL,
  bbox       TEXT    NOT NULL,   -- JSON-encoded points
  serial     TEXT,
  image_ref  TEXT
);
```

### 5.6 Persistence ↔ UI — D exposes, C consumes

C codes against this interface with an **in-memory fake** until D's SQLite lands. Stub-first.

```kotlin
interface FindingsRepository {
    suspend fun add(finding: Finding): Long
    suspend fun all(): List<Finding>
    fun stream(): Flow<List<Finding>>     // for live audit-log UI
    suspend fun export(): String          // CSV/JSON for "export via secure channel"
}
```

### 5.7 OCR seam (STRETCH) — defined now, built only if time allows

Wire the interface today; ship the stub. Real model only if MVP is done.

```kotlin
interface OcrManager {
    // crop = the pixels inside a classId=5 (label) detection
    suspend fun readText(crop: Bitmap): String   // stub returns "" 
}
```

Flow when enabled: detection `classId=5` → A crops the region → `OcrManager.readText()` →
fills `Finding.serial`.

---

## 6. How to work independently (stub-first playbook)

| You are… | You are blocked by… | Don't wait — do this |
|----------|--------------------|----------------------|
| **C (UI)** | B's model | Use the existing fake-detection `ModelManager`; build every screen against it. |
| **C (UI)** | D's SQLite | Code against `FindingsRepository` with an in-memory `FakeFindingsRepository`. |
| **A (Android)** | B's `.pte` | Keep the stub `ModelManager`; swap the body when the artifact arrives. Nothing else changes. |
| **D (Persist)** | the UI | Build + unit-test the repository headless; no UI needed. |
| **B (ML)** | everything | Fully isolated Python. Deliver `.pte` + output spec. Validate against `data/` images. |

**The golden rule:** if you need something that doesn't exist yet, write the *interface* (put it in
Section 5), ship a fake, and keep moving.

---

## 7. Branch & merge workflow

- **`main` is always green** — it builds and runs (with stubs) at all times.
- One feature branch per person: `name/stream` (e.g. `roshkins/android-core`).
- **Small, frequent PRs.** Rebase on `main` daily — don't let branches drift for two days.
- Changes to **shared files** (Section 3) go through Stream A as integrator.
- **Definition of done for a PR:** it builds, and it honors the Section 5 contracts.
- ⚠️ The current `roshkins/android` branch was cut from the *old* main and is missing the
  `setup_env.sh` fix on current main — **rebase it onto `main` before more work piles up.**

---

## 8. Integration milestones (2-day hackathon)

| When | Milestone | Looks like |
|------|-----------|-----------|
| **M0 — now** | Foundations frozen | This doc on `main`; class taxonomy frozen; `ui/` + `data/` packages stubbed; android scaffold merged to `main`. |
| **M1 — end of Day 1** | Skeleton walks | App runs: real UI + real SQLite + **fake** detections. Fully demo-able with stub data. |
| **M2 — Day 2 AM** | Real brains | B's `yolov8n_dcops_int8.pte` swapped into `ModelManager`; live detections on real racks. |
| **M3 — Day 2 PM** | Demo polish | Audit log populated, airplane-mode test passes, demo script rehearsed, latency numbers captured. |

The point of M1: **if the model slips, you still have a working demo.** The stub is insurance.

---

## 9. Open decisions (resolve at M0, ~15 min as a team)

1. **Folder name:** `android-app/` (actual) vs `android/app/` (docs). Pick one, fix README + CLAUDE.md.
2. **`.pte` delivery channel:** GitHub Release asset vs shared Drive. (Recommend Release.)
3. **Capture trigger:** auto-log every confident detection, or tap-to-confirm? Affects A + C + D.
4. **Min confidence threshold** to surface a detection (e.g. 0.5). Affects B + A.
5. **OCR go/no-go checkpoint:** decide at M2 whether there's time.

---

## 10. When you clone — quickstart by role

```bash
git clone <repo> && cd DC-Ops
source scripts/setup_env.sh        # Python venv + JDK + NDK + QNN env
git checkout -b <yourname>/<stream>
```

- **A (Android):** open `android-app/` in Android Studio → run on device. Align the stub
  `ModelManager` to `classes.yaml` (§5.1), then wire ExecuTorch AAR.
- **B (ML):** work in `models/`. Fine-tune YOLOv8n on `data/` to the 6 classes → quantize INT8 →
  `python models/export/export_yolo.py --soc_model SM8750`. Deliver `.pte` + output spec.
- **C (UI):** work in `ui/` + `overlay/` + `res/`. Build against the fake `ModelManager` and a
  `FakeFindingsRepository`. Don't touch `camera/` or `inference/`.
- **D (Data/Persist):** label images into `data/`; build `data/` package behind `FindingsRepository`
  (§5.6); own the demo script + keeping `main` green.
```

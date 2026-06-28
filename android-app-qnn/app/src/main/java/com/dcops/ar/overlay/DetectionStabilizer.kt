package com.dcops.ar.overlay

import android.graphics.PointF
import com.dcops.ar.inference.DetectionResult

/**
 * Per-object temporal stabilizer for noisy, near-threshold detections.
 *
 * The model (w8a16 YOLOv8n-seg) produces low, noisy confidences (~0.2-0.45) so the
 * per-frame detection SET changes frame-to-frame: real objects cross the confidence
 * threshold in and out, causing polygons to pop in/out ("flicker").
 *
 * This class fixes the appear/disappear flicker by tracking each object across frames
 * and applying:
 *   - IoU + class matching (in normalized 0..1 bbox space) to keep object identity,
 *   - a TTL hold so a momentarily-dropped object keeps drawing for a short window,
 *   - confidence hysteresis (Schmitt trigger: SHOW high, HIDE low) so scores hovering
 *     near the threshold don't toggle visibility,
 *   - minHits debounce so single-frame false detections never appear,
 *   - score EMA so the displayed percentage is calm.
 *
 * Shape (vertex) jitter is intentionally NOT smoothed: convex-hull vertex COUNTS vary
 * frame-to-frame, so per-vertex EMA is unreliable without resampling. Instead the most
 * recent matched polygon is kept and re-emitted for held (missed) frames.
 *
 * THREADING: not thread-safe by design. All access ([update] / [reset]) must happen on
 * a single thread (the UI thread in this app). No locking is used.
 */
class DetectionStabilizer {

    private companion object {
        // Hold a dropped track only briefly so the overlay tracks the camera with low
        // latency (was 450ms, which felt laggy). ~150ms still smooths 1-2 frame dropouts.
        const val TTL_MS = 150L
        // Secondary cap: remove after this many consecutive missed frames.
        const val MAX_MISSES = 4
        // Bbox-IoU gate for same-class association (lenient: noisy hulls wobble).
        const val IOU_MATCH = 0.30f
        // Normalized centroid-distance fallback when IoU == 0 (fast camera pans).
        const val CENTROID_DIST = 0.10f
        // Weight on the newest score for the displayed EMA percentage.
        const val EMA_ALPHA = 0.5f
        // Visibility is gated UPSTREAM by ModelManager.confThreshold (the Min-confidence
        // slider) in parseDetections, so detections reaching the stabilizer already passed
        // the user's threshold. The stabilizer must NOT impose its own absolute score floor
        // on top — that silently hid low-scoring models (e.g. RetinaNet, whose w8a16 scores
        // peak ~0.26) on the LIVE path while test mode (which bypasses the stabilizer) showed
        // them. Keep these at 0 so the stabilizer only adds temporal hold + MIN_HITS debounce.
        const val SHOW_SCORE = 0.0f
        const val HIDE_SCORE = 0.0f
        // Require this many matched frames before first showing a track.
        const val MIN_HITS = 2
        // Per-miss score decay so a vanished track eventually drops below HIDE_SCORE.
        const val MISS_DECAY = 0.85f
    }

    /** Axis-aligned bounding box in normalized 0..1 coords. */
    private class Box(val left: Float, val top: Float, val right: Float, val bottom: Float) {
        val cx: Float get() = (left + right) * 0.5f
        val cy: Float get() = (top + bottom) * 0.5f
        val area: Float get() = (right - left).coerceAtLeast(0f) * (bottom - top).coerceAtLeast(0f)
    }

    private class Track(
        var classId: Int,
        var label: String,
        var polygon: List<PointF>,
        var box: Box,
        var emaScore: Float,
        var lastSeenMs: Long,
        var hits: Int,
        var misses: Int,
        var visible: Boolean,
        val id: Long
    )

    private val tracks = ArrayList<Track>(64)
    private var nextId = 0L

    /** Clears all tracked state. Call when switching modes so nothing bleeds across. */
    fun reset() {
        tracks.clear()
    }

    /**
     * Stabilize one frame of (already confidence-filtered) detections.
     *
     * @param raw   detections for this frame; polygons normalized 0..1.
     * @param nowMs monotonic timestamp, e.g. android.os.SystemClock.uptimeMillis().
     * @return the list of detections that should be drawn this frame (held/debounced).
     */
    fun update(raw: List<DetectionResult>, nowMs: Long): List<DetectionResult> {
        // Precompute bboxes once per incoming detection.
        val detBoxes = ArrayList<Box>(raw.size)
        for (d in raw) detBoxes.add(bboxOf(d.polygon))

        val detClaimed = BooleanArray(raw.size)
        val trackMatched = BooleanArray(tracks.size)

        // ---- MATCH: greedy by score (desc), class-gated, IoU-gated ----
        // Build candidate (detIdx, trackIdx, iou) pairs.
        val pairs = ArrayList<IntArray>() // [detIdx, trackIdx]
        val pairIou = ArrayList<Float>()
        for (di in raw.indices) {
            val dBox = detBoxes[di]
            for (ti in tracks.indices) {
                val t = tracks[ti]
                if (t.classId != raw[di].classId) continue
                val iou = iou(dBox, t.box)
                if (iou >= IOU_MATCH) {
                    pairs.add(intArrayOf(di, ti))
                    pairIou.add(iou)
                }
            }
        }
        // Sort candidate pairs by det score desc, tiebreak higher IoU.
        val order = pairs.indices.sortedWith(compareByDescending<Int> { raw[pairs[it][0]].score }
            .thenByDescending { pairIou[it] })
        for (oi in order) {
            val di = pairs[oi][0]
            val ti = pairs[oi][1]
            if (detClaimed[di] || trackMatched[ti]) continue
            detClaimed[di] = true
            trackMatched[ti] = true
            updateMatched(tracks[ti], raw[di], detBoxes[di], nowMs)
        }

        // ---- Centroid fallback for still-unmatched dets (fast motion) ----
        for (di in raw.indices) {
            if (detClaimed[di]) continue
            val dBox = detBoxes[di]
            var bestTi = -1
            var bestDist = CENTROID_DIST
            for (ti in tracks.indices) {
                if (trackMatched[ti]) continue
                val t = tracks[ti]
                if (t.classId != raw[di].classId) continue
                val dist = centerDist(dBox, t.box)
                if (dist < bestDist) {
                    bestDist = dist
                    bestTi = ti
                }
            }
            if (bestTi >= 0) {
                detClaimed[di] = true
                trackMatched[bestTi] = true
                updateMatched(tracks[bestTi], raw[di], dBox, nowMs)
            }
        }

        // ---- AGE unmatched tracks ----
        // Done BEFORE spawning so trackMatched indices stay aligned with the pre-spawn
        // track set, and so freshly spawned tracks aren't aged on their first frame.
        for (ti in trackMatched.indices) {
            if (trackMatched[ti]) continue
            val t = tracks[ti]
            t.misses += 1
            t.emaScore *= MISS_DECAY
        }

        // ---- SPAWN new tracks for unmatched dets ----
        for (di in raw.indices) {
            if (detClaimed[di]) continue
            val d = raw[di]
            tracks.add(
                Track(
                    classId = d.classId,
                    label = d.label,
                    polygon = d.polygon,
                    box = detBoxes[di],
                    emaScore = d.score,
                    lastSeenMs = nowMs,
                    hits = 1,
                    misses = 0,
                    visible = false,
                    id = nextId++
                )
            )
        }

        // ---- PRUNE + HYSTERESIS + EMIT ----
        val out = ArrayList<DetectionResult>(tracks.size)
        val it = tracks.iterator()
        while (it.hasNext()) {
            val t = it.next()
            val aged = nowMs - t.lastSeenMs
            if (aged > TTL_MS || t.misses > MAX_MISSES) {
                it.remove()
                continue
            }
            // Hysteresis (Schmitt trigger).
            if (!t.visible) {
                if (t.emaScore >= SHOW_SCORE && t.hits >= MIN_HITS) t.visible = true
            } else {
                if (t.emaScore < HIDE_SCORE) t.visible = false
            }
            if (t.visible) {
                out.add(DetectionResult(t.label, t.emaScore, t.polygon, t.classId))
            }
        }
        return out
    }

    private fun updateMatched(t: Track, d: DetectionResult, box: Box, nowMs: Long) {
        t.polygon = d.polygon
        t.box = box
        t.label = d.label
        t.emaScore = EMA_ALPHA * d.score + (1f - EMA_ALPHA) * t.emaScore
        t.lastSeenMs = nowMs
        t.hits += 1
        t.misses = 0
    }

    private fun bboxOf(poly: List<PointF>): Box {
        if (poly.isEmpty()) return Box(0f, 0f, 0f, 0f)
        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var maxY = -Float.MAX_VALUE
        for (p in poly) {
            if (p.x < minX) minX = p.x
            if (p.y < minY) minY = p.y
            if (p.x > maxX) maxX = p.x
            if (p.y > maxY) maxY = p.y
        }
        return Box(minX, minY, maxX, maxY)
    }

    private fun iou(a: Box, b: Box): Float {
        val interLeft = maxOf(a.left, b.left)
        val interTop = maxOf(a.top, b.top)
        val interRight = minOf(a.right, b.right)
        val interBottom = minOf(a.bottom, b.bottom)
        val interW = interRight - interLeft
        val interH = interBottom - interTop
        if (interW <= 0f || interH <= 0f) return 0f
        val inter = interW * interH
        val union = a.area + b.area - inter
        if (union <= 0f) return 0f
        return inter / union
    }

    private fun centerDist(a: Box, b: Box): Float {
        val dx = a.cx - b.cx
        val dy = a.cy - b.cy
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }
}

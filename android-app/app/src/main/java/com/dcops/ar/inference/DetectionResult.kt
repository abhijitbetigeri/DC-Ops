package com.dcops.ar.inference

import android.graphics.PointF
import android.graphics.RectF

/**
 * A single detection produced by the ML model.
 *
 * @property label    Human-readable class name (from [DcClass.displayName]).
 * @property score    Confidence score in [0, 1].
 * @property polygon  List of [PointF] vertices in **normalized image coordinates**
 *                    (0.0–1.0). The overlay maps these to view pixels at draw time.
 *                    A bounding-box detection will have exactly 4 points; a true
 *                    polygon mask may have more.
 * @property classId  Integer class index from the model — see [DcClass].
 */
data class DetectionResult(
    val label: String,
    val score: Float,
    val polygon: List<PointF>,
    val classId: Int = -1
) {
    /** Returns true if this detection has at least 3 vertices (a real polygon). */
    fun isPolygon(): Boolean = polygon.size >= 3

    /** The DC-Ops taxonomy class for this detection, or null if unrecognized. */
    val dcClass: DcClass? get() = DcClass.fromId(classId)

    /** Axis-aligned bounding box (still normalized 0..1) enclosing the polygon. */
    fun boundingBox(): RectF {
        if (polygon.isEmpty()) return RectF()
        var left = Float.MAX_VALUE
        var top = Float.MAX_VALUE
        var right = -Float.MAX_VALUE
        var bottom = -Float.MAX_VALUE
        for (p in polygon) {
            if (p.x < left) left = p.x
            if (p.x > right) right = p.x
            if (p.y < top) top = p.y
            if (p.y > bottom) bottom = p.y
        }
        return RectF(left, top, right, bottom)
    }
}

package com.dcops.ar.inference

import android.graphics.PointF

/**
 * A single detection produced by the ML model.
 *
 * @property label    Human-readable class name (e.g. "LED-green", "cable", "label").
 * @property score    Confidence score in [0, 1].
 * @property polygon  List of [PointF] vertices in **normalized image coordinates**
 *                    (0.0–1.0). The overlay maps these to view pixels at draw time.
 *                    A bounding-box detection will have exactly 4 points; a true
 *                    polygon mask may have more.
 * @property classId  Integer class index from the model (optional).
 */
data class DetectionResult(
    val label: String,
    val score: Float,
    val polygon: List<PointF>,
    val classId: Int = -1
) {
    /**
     * Returns true if this detection has at least 3 vertices (a real polygon).
     */
    fun isPolygon(): Boolean = polygon.size >= 3
}
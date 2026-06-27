package com.dcops.ar.data

import android.graphics.PointF
import com.dcops.ar.inference.DcClass
import com.dcops.ar.inference.DetectionResult

/**
 * An audit-log record — written when a technician captures/confirms a detection.
 * Matches ARCHITECTURE.md §5.5. This is the contract the persistence layer
 * stores and the UI reads.
 *
 * @property bbox normalized (0..1) polygon, carried over from [DetectionResult].
 * @property serial OCR result (stretch goal, may be null).
 * @property imageRef optional path to a saved frame.
 */
data class Finding(
    val id: Long = 0,
    val timestampMs: Long,
    val classId: Int,
    val label: String,
    val score: Float,
    val bbox: List<PointF>,
    val serial: String? = null,
    val imageRef: String? = null
) {
    val dcClass: DcClass? get() = DcClass.fromId(classId)

    companion object {
        /** Build a [Finding] from a live [DetectionResult] at [timestampMs]. */
        fun from(d: DetectionResult, timestampMs: Long, serial: String? = null) = Finding(
            timestampMs = timestampMs,
            classId = d.classId,
            label = d.label,
            score = d.score,
            bbox = d.polygon,
            serial = serial
        )
    }
}

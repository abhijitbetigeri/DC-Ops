package com.dcops.ar.overlay

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.util.AttributeSet
import android.view.View
import com.dcops.ar.inference.DetectionResult

/**
 * Custom [View] that draws detection polygons on top of the live camera preview.
 *
 * Detection polygons arrive as a list of [DetectionResult] whose vertex
 * coordinates are **normalized** (0.0–1.0).  At draw time these are scaled to
 * the actual pixel dimensions of this view, producing a real-time AR overlay.
 *
 * Color is chosen based on the detection label so different classes are
 * visually distinguishable (LED states, cables, labels, etc.).
 */
class PolygonOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<DetectionResult> = emptyList()

    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        alpha = 60
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 36f
        isFakeBoldText = true
        setShadowLayer(4f, 1f, 1f, Color.BLACK)
    }

    private val textBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#AA000000")
        style = Paint.Style.FILL
    }

    /** Update detections and trigger a redraw.  Call on the main thread. */
    fun updateDetections(newDetections: List<DetectionResult>) {
        detections = newDetections
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0 || h <= 0) return

        for (det in detections) {
            if (det.polygon.isEmpty()) continue

            // Prefer the frozen taxonomy color (by classId); fall back to label match.
            val color = det.dcClass?.color ?: colorForLabel(det.label)

            // Build the polygon path in pixel coordinates
            val path = Path()
            val first = det.polygon[0]
            path.moveTo(first.x * w, first.y * h)
            for (i in 1 until det.polygon.size) {
                path.lineTo(det.polygon[i].x * w, det.polygon[i].y * h)
            }
            path.close()

            // Fill (semi-transparent)
            fillPaint.color = color
            canvas.drawPath(path, fillPaint)

            // Stroke
            strokePaint.color = color
            canvas.drawPath(path, strokePaint)

            // Draw vertices
            strokePaint.style = Paint.Style.FILL
            for (p in det.polygon) {
                canvas.drawCircle(p.x * w, p.y * h, 5f, strokePaint)
            }
            strokePaint.style = Paint.Style.STROKE

            // Draw label text at the top-most vertex
            val labelPoint = det.polygon.minByOrNull { it.y } ?: det.polygon[0]
            val labelX = labelPoint.x * w
            val labelY = labelPoint.y * h - 12f
            val labelText = "${det.label}  ${(det.score * 100).toInt()}%"
            val textWidth = textPaint.measureText(labelText)
            val textHeight = textPaint.textSize

            canvas.drawRect(
                labelX - 6f,
                labelY - textHeight,
                labelX + textWidth + 6f,
                labelY + 6f,
                textBgPaint
            )
            canvas.drawText(labelText, labelX, labelY, textPaint)
        }
    }

    /**
     * Map a detection label to an overlay color.
     * Extend this as new classes are added.
     */
    private fun colorForLabel(label: String): Int {
        return when {
            label.contains("green", ignoreCase = true) -> Color.parseColor("#00E676")
            label.contains("amber", ignoreCase = true) -> Color.parseColor("#FFC107")
            label.contains("red", ignoreCase = true)   -> Color.parseColor("#FF1744")
            label.contains("cable", ignoreCase = true) -> Color.parseColor("#2979FF")
            label.contains("label", ignoreCase = true) -> Color.parseColor("#E040FB")
            else -> Color.parseColor("#FFFFFF")
        }
    }
}
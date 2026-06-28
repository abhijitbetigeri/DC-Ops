package com.dcops.ar.overlay

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.CornerPathEffect
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View
import com.dcops.ar.inference.DetectionResult
import com.dcops.ar.ui.DetectionMissionStyle
import com.dcops.ar.ui.DetectionUrgency

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
    private var focusDetection: DetectionResult? = null

    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
        pathEffect = CornerPathEffect(18f)
    }

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 34f
        isFakeBoldText = true
        setShadowLayer(4f, 1f, 1f, Color.BLACK)
    }

    private val textBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#AA000000")
        style = Paint.Style.FILL
    }

    /** Update detections and trigger a redraw.  Call on the main thread. */
    fun updateDetections(
        newDetections: List<DetectionResult>,
        focus: DetectionResult? = null
    ) {
        detections = newDetections
        focusDetection = focus
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0 || h <= 0) return

        for (det in detections) {
            if (det.polygon.isEmpty()) continue

            val urgency = DetectionMissionStyle.urgencyFor(det)
            val color = colorFor(det, urgency)
            val isFocus = det == focusDetection
            val isLowConfidence = det.score < 0.60f

            val path = Path()
            val first = det.polygon[0]
            path.moveTo(first.x * w, first.y * h)
            for (i in 1 until det.polygon.size) {
                path.lineTo(det.polygon[i].x * w, det.polygon[i].y * h)
            }
            path.close()

            fillPaint.color = color
            fillPaint.alpha = when {
                isFocus -> 76
                urgency == DetectionUrgency.CRITICAL -> 64
                else -> 42
            }
            canvas.drawPath(path, fillPaint)

            strokePaint.color = color
            strokePaint.strokeWidth = if (isFocus) 8f else 4f
            strokePaint.pathEffect = when {
                isFocus -> CornerPathEffect(24f)
                isLowConfidence || urgency == DetectionUrgency.REVIEW -> DashPathEffect(
                    floatArrayOf(18f, 14f),
                    0f
                )
                else -> CornerPathEffect(18f)
            }
            canvas.drawPath(path, strokePaint)

            strokePaint.style = Paint.Style.FILL
            for (p in det.polygon) {
                canvas.drawCircle(p.x * w, p.y * h, if (isFocus) 7f else 5f, strokePaint)
            }
            strokePaint.style = Paint.Style.STROKE

            val labelPoint = det.polygon.minByOrNull { it.y } ?: det.polygon[0]
            val labelX = labelPoint.x * w
            val labelY = labelPoint.y * h - 12f
            val labelText = "${det.label.uppercase()}  ${(det.score * 100).toInt()}%"
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

    companion object {
        private const val HEALTHY_COLOR = "#2BE38A"
        private const val REVIEW_COLOR = "#FFB020"
        private const val CRITICAL_COLOR = "#FF5A5F"
        private const val FALLBACK_COLOR = "#62D4FF"
    }

    private fun colorFor(det: DetectionResult, urgency: DetectionUrgency): Int {
        return when (urgency) {
            DetectionUrgency.HEALTHY -> {
                if (det.classId == 8) Color.parseColor(FALLBACK_COLOR) else Color.parseColor(HEALTHY_COLOR)
            }
            DetectionUrgency.REVIEW -> Color.parseColor(REVIEW_COLOR)
            DetectionUrgency.CRITICAL -> Color.parseColor(CRITICAL_COLOR)
        }
    }
}
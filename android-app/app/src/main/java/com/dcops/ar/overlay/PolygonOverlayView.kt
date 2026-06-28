package com.dcops.ar.overlay

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.CornerPathEffect
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.os.SystemClock
import android.util.AttributeSet
import android.view.View
import com.dcops.ar.inference.DetectionResult
import com.dcops.ar.ui.DetectionMissionStyle
import com.dcops.ar.ui.DetectionUrgency
import kotlin.math.sin

data class CableMatchVisualization(
    val cableDetection: DetectionResult?,
    val portDetection: DetectionResult?,
    val cableLabel: String,
    val portLabel: String,
    val activeColor: Int
)

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
    private var cableMatchVisualization: CableMatchVisualization? = null

    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
        pathEffect = CornerPathEffect(18f)
    }

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val matchPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 9f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        setShadowLayer(14f, 0f, 0f, Color.BLACK)
    }

    private val pulsePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 7f
    }

    /** Update detections and trigger a redraw.  Call on the main thread. */
    fun updateDetections(
        newDetections: List<DetectionResult>,
        focus: DetectionResult? = null,
        cableMatch: CableMatchVisualization? = null
    ) {
        detections = newDetections
        focusDetection = focus
        cableMatchVisualization = cableMatch
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
        }

        drawCableMatch(canvas, w, h)
    }

    private fun drawCableMatch(canvas: Canvas, width: Float, height: Float) {
        val match = cableMatchVisualization ?: return
        val port = match.portDetection ?: return
        val portCenter = centerOf(port, width, height) ?: return
        val pulse = ((sin(SystemClock.uptimeMillis() / 180.0) + 1.0) / 2.0).toFloat()
        val pulseRadius = 30f + pulse * 22f

        pulsePaint.color = match.activeColor
        pulsePaint.alpha = 155 + (pulse * 85f).toInt()
        canvas.drawCircle(portCenter.x, portCenter.y, pulseRadius, pulsePaint)
        canvas.drawCircle(portCenter.x, portCenter.y, pulseRadius + 18f, pulsePaint)

        val cableCenter = match.cableDetection?.let { centerOf(it, width, height) } ?: return
        matchPaint.color = match.activeColor
        matchPaint.alpha = 230
        canvas.drawLine(cableCenter.x, cableCenter.y, portCenter.x, portCenter.y, matchPaint)

        fillPaint.color = match.activeColor
        fillPaint.alpha = 210
        canvas.drawCircle(cableCenter.x, cableCenter.y, 13f, fillPaint)
        canvas.drawCircle(portCenter.x, portCenter.y, 13f, fillPaint)
    }

    private fun centerOf(detection: DetectionResult, width: Float, height: Float): PointF? {
        if (detection.polygon.isEmpty()) return null
        val centerX = detection.polygon.map { it.x }.average().toFloat() * width
        val centerY = detection.polygon.map { it.y }.average().toFloat() * height
        return PointF(centerX, centerY)
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
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
import com.dcops.ar.inference.ModelManager

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

    /** Top safe-area inset (status bar height, px). Set by the activity from window insets. */
    var topInset: Float = 0f
        set(value) { field = value; invalidate() }

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

            val color = colorForClass(det.classId)

            // Build the polygon path in pixel coordinates
            val path = Path()
            val first = det.polygon[0]
            path.moveTo(first.x * w, first.y * h)
            for (i in 1 until det.polygon.size) {
                path.lineTo(det.polygon[i].x * w, det.polygon[i].y * h)
            }
            path.close()

            // Fill (translucent) — set alpha AFTER color, since setting color resets alpha to
            // the opaque value baked into CLASS_COLORS. ~90/255 lets the image and overlapping
            // detections show through.
            fillPaint.color = color
            fillPaint.alpha = 90
            canvas.drawPath(path, fillPaint)

            // Stroke (mostly opaque so the outline stays readable over the fill)
            strokePaint.color = color
            strokePaint.alpha = 220
            canvas.drawPath(path, strokePaint)

            // Draw vertices
            strokePaint.style = Paint.Style.FILL
            for (p in det.polygon) {
                canvas.drawCircle(p.x * w, p.y * h, 5f, strokePaint)
            }
            strokePaint.style = Paint.Style.STROKE

            // Draw label text near the top-most vertex, but CLAMP it so the box is always
            // fully on-screen — even when the polygon fills (or exceeds) the whole view.
            val labelPoint = det.polygon.minByOrNull { it.y } ?: det.polygon[0]
            val labelText = "${det.label}  ${(det.score * 100).toInt()}%"
            val textWidth = textPaint.measureText(labelText)
            val textHeight = textPaint.textSize
            val margin = 8f
            val topSafe = topInset + textHeight + 8f   // keep clear of the status bar / screen top

            val labelX = (labelPoint.x * w).coerceIn(margin, (w - textWidth - margin).coerceAtLeast(margin))
            var baseline = labelPoint.y * h - 12f
            if (baseline < topSafe) baseline = topSafe                 // pushed below top edge
            if (baseline + 6f > h - margin) baseline = h - margin - 6f // not under bottom edge

            canvas.drawRect(
                labelX - 6f,
                baseline - textHeight,
                labelX + textWidth + 6f,
                baseline + 6f,
                textBgPaint
            )
            canvas.drawText(labelText, labelX, baseline, textPaint)
        }
    }

    companion object {
        // One distinct color per DC class for visual separation
        private val CLASS_COLORS = intArrayOf(
            Color.parseColor("#4CAF50"),  //  0 server rack      — green
            Color.parseColor("#FF9800"),  //  1 compute tray     — orange
            Color.parseColor("#9C27B0"),  //  2 NVLink switch    — purple
            Color.parseColor("#2196F3"),  //  3 network switch   — blue
            Color.parseColor("#F44336"),  //  4 power shelf      — red
            Color.parseColor("#00BCD4"),  //  5 cable            — cyan
            Color.parseColor("#FFEB3B"),  //  6 network port     — yellow
            Color.parseColor("#00E676"),  //  7 LED indicator    — bright green
            Color.parseColor("#E040FB"),  //  8 label            — magenta
            Color.parseColor("#795548"),  //  9 fan              — brown
            Color.parseColor("#03A9F4"),  // 10 cooling manifold — light blue
            Color.parseColor("#FF5722"),  // 11 cable cartridge  — deep orange
            Color.parseColor("#FFC107"),  // 12 power connector  — amber
            Color.parseColor("#8BC34A"),  // 13 drive bay        — light green
            Color.parseColor("#607D8B"),  // 14 management port  — blue grey
            Color.parseColor("#FF1744"),  // 15 DPU              — red accent
        )
    }

    private fun colorForClass(classId: Int): Int {
        if (classId in CLASS_COLORS.indices) return CLASS_COLORS[classId]
        return Color.WHITE
    }
}
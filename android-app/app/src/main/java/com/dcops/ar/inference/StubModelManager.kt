package com.dcops.ar.inference

import android.graphics.PointF
import androidx.camera.core.ImageProxy

/**
 * Fake [ModelManager] that returns deterministic, animated detections so the
 * entire camera → overlay → capture → audit-log pipeline is demo-able **before
 * the real YOLO model exists**.
 *
 * Class IDs here match the frozen taxonomy in [DcClass] (ARCHITECTURE.md §5.1):
 *   - a pulsing rectangle  → LED_GREEN (0)
 *   - a hexagon            → LABEL (5)
 *   - a pentagon           → CABLE (4)
 *
 * Replace with [ExecuTorchModelManager] once `yolov8n_dcops_int8.pte` ships.
 */
class StubModelManager : ModelManager {

    private var frameCount = 0L

    override fun init(onReady: (Boolean) -> Unit) {
        Thread {
            Thread.sleep(500) // simulate load latency
            onReady(true)
        }.start()
    }

    override fun processFrame(imageProxy: ImageProxy, onResult: (List<DetectionResult>) -> Unit) {
        frameCount++

        // Pulse so the overlay is visibly alive during stub testing.
        val t = (frameCount % 60).toFloat() / 60f
        val cx = 0.5f
        val cy = 0.5f
        val w = 0.18f + 0.02f * kotlin.math.sin(t * 2 * Math.PI.toFloat())
        val h = 0.12f + 0.02f * kotlin.math.cos(t * 2 * Math.PI.toFloat())

        val detections = mutableListOf<DetectionResult>()

        // LED (green) — rectangle near center
        detections += detection(
            DcClass.LED_GREEN, 0.92f,
            listOf(
                PointF(cx - w, cy - h),
                PointF(cx + w, cy - h),
                PointF(cx + w, cy + h),
                PointF(cx - w, cy + h)
            )
        )

        // Label / serial region — hexagon upper-left
        detections += detection(DcClass.LABEL, 0.78f, regularPolygon(0.25f, 0.30f, 0.08f, 6, -30.0))

        // Cable — pentagon lower-right
        detections += detection(DcClass.CABLE, 0.85f, regularPolygon(0.72f, 0.68f, 0.07f, 5, -90.0))

        // Close BEFORE the simulated delay so the pipeline keeps flowing.
        imageProxy.close()
        Thread.sleep(30) // simulate inference latency

        onResult(detections)
    }

    override fun shutdown() { /* nothing to release for the stub */ }

    private fun detection(cls: DcClass, score: Float, polygon: List<PointF>) =
        DetectionResult(label = cls.displayName, score = score, polygon = polygon, classId = cls.id)

    private fun regularPolygon(
        cx: Float, cy: Float, r: Float, sides: Int, startDeg: Double
    ): List<PointF> = (0 until sides).map { i ->
        val angle = Math.toRadians(startDeg + i * (360.0 / sides))
        PointF((cx + r * kotlin.math.cos(angle)).toFloat(), (cy + r * kotlin.math.sin(angle)).toFloat())
    }
}

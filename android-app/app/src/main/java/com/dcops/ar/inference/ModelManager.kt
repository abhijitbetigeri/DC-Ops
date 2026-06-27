package com.dcops.ar.inference

import android.graphics.PointF
import androidx.camera.core.ImageProxy
import kotlin.random.Random

/**
 * Stub model manager.
 *
 * In the real implementation this will:
 *  1. Load a .pte model (YOLOv8n or OCR) via the ExecuTorch AAR runtime.
 *  2. Convert each [ImageProxy] to a FloatBuffer / ByteBuffer in the format
 *     expected by the model (e.g. NCHW float32, 640×640, normalized).
 *  3. Run forward inference on the Snapdragon NPU (QNN backend).
 *  4. Parse the raw output tensor into a list of [DetectionResult] polygons.
 *
 * For now, [processFrame] returns a deterministic set of sample polygons so the
 * overlay rendering pipeline can be validated end-to-end.
 *
 * When you are ready to wire in the real model, replace the body of
 * [init] and [processFrame] — the rest of the app does not need to change.
 */
class ModelManager {

    private var isReady = false
    private var frameCount = 0L

    /**
     * Simulate model loading.  In production this would load the .pte file
     * and initialize the QNN backend.
     *
     * @param onReady Called on a background thread when "loading" completes.
     */
    fun init(onReady: (Boolean) -> Unit) {
        // TODO: Load ExecuTorch module from assets (e.g. "yolov8n_int8.pte")
        //   val module = Module.load(assetFilePath(context, "yolov8n_int8.pte"))
        //   ...configure QNN backend...
        Thread {
            Thread.sleep(500) // simulate load latency
            isReady = true
            onReady(true)
        }.start()
    }

    /**
     * Process a single camera frame.
     *
     * @param imageProxy  The frame from CameraX ImageAnalysis.  The callee
     *                    MUST call [ImageProxy.close] when done.
     * @param onResult    Callback invoked with detection results (on a
     *                    background thread).  The results are in normalized
     *                    image coordinates (0.0–1.0).
     */
    fun processFrame(imageProxy: ImageProxy, onResult: (List<DetectionResult>) -> Unit) {
        // ── STUB: generate sample detections ──────────────────────────────
        //
        // Replace this entire block with:
        //   1. Convert imageProxy → input tensor (resize to model input size,
        //      normalize, NCHW layout).
        //   2. Run forward:  val outputs = module.forward(method, inputs)
        //   3. Parse outputs → List<DetectionResult>

        frameCount++

        // Pulsing effect so the overlay is visibly animated during stub testing
        val t = (frameCount % 60).toFloat() / 60f
        val cx = 0.5f
        val cy = 0.5f
        val w = 0.18f + 0.02f * kotlin.math.sin(t * 2 * Math.PI.toFloat())
        val h = 0.12f + 0.02f * kotlin.math.cos(t * 2 * Math.PI.toFloat())

        val sampleDetections = mutableListOf<DetectionResult>()

        // Detection 1 — a rectangle near center (simulates LED detection)
        sampleDetections.add(
            DetectionResult(
                label = "LED-green",
                score = 0.92f,
                polygon = listOf(
                    PointF(cx - w, cy - h),
                    PointF(cx + w, cy - h),
                    PointF(cx + w, cy + h),
                    PointF(cx - w, cy + h)
                ),
                classId = 0
            )
        )

        // Detection 2 — a hexagon offset to upper-left (simulates a polygon mask)
        val hexCenterX = 0.25f
        val hexCenterY = 0.30f
        val hexR = 0.08f
        val hexPoints = (0 until 6).map { i ->
            val angle = (i * 60).toDouble() - 30.0
            PointF(
                (hexCenterX + (hexR * kotlin.math.cos(Math.toRadians(angle)))).toFloat(),
                (hexCenterY + (hexR * kotlin.math.sin(Math.toRadians(angle)))).toFloat()
            )
        }
        sampleDetections.add(
            DetectionResult(
                label = "label",
                score = 0.78f,
                polygon = hexPoints,
                classId = 1
            )
        )

        // Detection 3 — a pentagon lower-right (simulates cable region)
        val pentCX = 0.72f
        val pentCY = 0.68f
        val pentR = 0.07f
        val pentPoints = (0 until 5).map { i ->
            val angle = (i * 72).toDouble() - 90.0
            PointF(
                (pentCX + (pentR * kotlin.math.cos(Math.toRadians(angle)))).toFloat(),
                (pentCY + (pentR * kotlin.math.sin(Math.toRadians(angle)))).toFloat()
            )
        }
        sampleDetections.add(
            DetectionResult(
                label = "cable",
                score = 0.85f,
                polygon = pentPoints,
                classId = 2
            )
        )

        // IMPORTANT: close the ImageProxy so CameraX can deliver the next frame
        imageProxy.close()

        // Simulate a small inference delay
        Thread.sleep(30)

        onResult(sampleDetections)

        // ── END STUB ──────────────────────────────────────────────────────
    }

    fun shutdown() {
        // TODO: release ExecuTorch module / QNN resources
        isReady = false
    }
}
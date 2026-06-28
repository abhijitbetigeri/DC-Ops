package com.dcops.ar.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import kotlin.math.exp

class ModelManager {

    companion object {
        const val MODEL_FILENAME = "dc_ops_yolov8n_seg.pte"
        const val QNN_MODEL_FILENAME = "dc_ops_retinanet_qnn.pte"
        const val INPUT_SIZE = 640
        const val CONF_THRESHOLD = 0.25f
        const val IOU_THRESHOLD = 0.45f
        const val MAX_DETECTIONS = 50

        val DC_CLASSES = arrayOf(
            "server rack", "compute tray", "NVLink switch tray", "network switch",
            "power shelf", "cable", "network port", "LED indicator",
            "label", "fan", "cooling manifold", "cable cartridge",
            "power connector", "drive bay", "management port", "DPU"
        )

        // ImageNet normalization for RetinaNet
        val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private var module: Any? = null
    private var isReady = false
    private var isRetinaNet = false
    private val inputBuffer = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)

    fun init(context: Context, onReady: (Boolean) -> Unit) {
        Thread {
            try {
                val modelPath = assetFilePath(context, MODEL_FILENAME)
                module = loadExecuTorchModule(modelPath)
                isRetinaNet = false
                isReady = true
                onReady(true)
            } catch (e: Exception) {
                e.printStackTrace()
                isReady = false
                onReady(false)
            }
        }.start()
    }

    fun switchModel(context: Context, modelFile: String, onReady: (Boolean) -> Unit) {
        isReady = false
        Thread {
            try {
                module?.let { m ->
                    m.javaClass.methods.firstOrNull { it.name == "destroy" && it.parameterTypes.isEmpty() }?.invoke(m)
                }
                val modelPath = assetFilePath(context, modelFile)
                module = loadExecuTorchModule(modelPath)
                isRetinaNet = modelFile.contains("retinanet")
                isReady = true
                onReady(true)
            } catch (e: Exception) {
                e.printStackTrace()
                isReady = false
                onReady(false)
            }
        }.start()
    }

    fun processFrame(imageProxy: ImageProxy, onResult: (List<DetectionResult>) -> Unit) {
        val loadedModule = module ?: run { imageProxy.close(); return }
        if (!isReady) { imageProxy.close(); return }

        try {
            val bitmap = imageProxyToBitmap(imageProxy)
            val scaled = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
            val inputTensor = if (isRetinaNet) {
                bitmapToEValueNormalized(scaled)
            } else {
                bitmapToEValue(scaled)
            }

            val outputs = runModuleForward(loadedModule, inputTensor)
            val results = if (isRetinaNet) {
                parseRetinaNetOutput(outputs)
            } else {
                parseYoloOutput(outputs)
            }
            onResult(results)
        } catch (e: Exception) {
            e.printStackTrace()
            onResult(emptyList())
        } finally {
            imageProxy.close()
        }
    }

    // --- Input preprocessing ---

    private fun bitmapToEValue(bitmap: Bitmap): Any {
        inputBuffer.rewind()
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (pixel in pixels) inputBuffer.put(((pixel shr 16) and 0xFF) / 255.0f)
        for (pixel in pixels) inputBuffer.put(((pixel shr 8) and 0xFF) / 255.0f)
        for (pixel in pixels) inputBuffer.put((pixel and 0xFF) / 255.0f)
        inputBuffer.rewind()
        val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        return wrapTensorInEValue(createTensorFromBlob(inputBuffer, shape))
    }

    private fun bitmapToEValueNormalized(bitmap: Bitmap): Any {
        inputBuffer.rewind()
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        // R channel normalized
        for (pixel in pixels) inputBuffer.put((((pixel shr 16) and 0xFF) / 255.0f - MEAN[0]) / STD[0])
        // G channel normalized
        for (pixel in pixels) inputBuffer.put((((pixel shr 8) and 0xFF) / 255.0f - MEAN[1]) / STD[1])
        // B channel normalized
        for (pixel in pixels) inputBuffer.put(((pixel and 0xFF) / 255.0f - MEAN[2]) / STD[2])
        inputBuffer.rewind()
        val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        return wrapTensorInEValue(createTensorFromBlob(inputBuffer, shape))
    }

    // --- RetinaNet output parsing ---
    // RetinaNet outputs 2 tensors:
    //   [0] cls_logits: [1, num_anchors, num_classes+1]
    //   [1] bbox_regression: [1, num_anchors, 4]

    private fun parseRetinaNetOutput(outputs: Array<*>): List<DetectionResult> {
        if (outputs.size < 2) return emptyList()

        val clsTensor = toTensor(outputs[0] ?: return emptyList())
        val bboxTensor = toTensor(outputs[1] ?: return emptyList())

        val clsData = tensorDataAsFloatArray(clsTensor)
        val bboxData = tensorDataAsFloatArray(bboxTensor)
        val clsShape = tensorShape(clsTensor)
        val bboxShape = tensorShape(bboxTensor)

        // Determine layout from shapes
        val numAnchors: Int
        val numClasses: Int
        val clsStride: Int
        val bboxStride: Int

        if (clsShape.size == 3) {
            // [1, num_anchors, num_classes] or [1, num_classes, num_anchors]
            if (clsShape[2] <= 20) {
                // [1, num_anchors, num_classes]
                numAnchors = clsShape[1].toInt()
                numClasses = clsShape[2].toInt()
                clsStride = numClasses // row-major: anchor * numClasses + class
            } else {
                // [1, num_classes, num_anchors]
                numClasses = clsShape[1].toInt()
                numAnchors = clsShape[2].toInt()
                clsStride = numAnchors // col-major: class * numAnchors + anchor
            }
        } else if (clsShape.size == 2) {
            numAnchors = clsShape[0].toInt()
            numClasses = clsShape[1].toInt()
            clsStride = numClasses
        } else {
            return emptyList()
        }

        val bboxPerAnchor = if (bboxShape.size == 3) {
            if (bboxShape[2] == 4L) true else false // [1, anchors, 4] vs [1, 4, anchors]
        } else true

        val results = mutableListOf<DetectionResult>()
        val scale = 1.0f / INPUT_SIZE

        for (a in 0 until numAnchors) {
            // Find best class (skip background class 0)
            var bestClass = -1
            var bestScore = 0f

            for (c in 1 until minOf(numClasses, DC_CLASSES.size + 1)) {
                val rawScore = if (clsStride == numClasses) {
                    clsData[a * numClasses + c]
                } else {
                    clsData[c * numAnchors + a]
                }
                val score = sigmoid(rawScore)
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c - 1  // Convert to 0-indexed DC class
                }
            }

            if (bestScore < CONF_THRESHOLD || bestClass < 0 || bestClass >= DC_CLASSES.size) continue

            // Get bbox
            val x1: Float; val y1: Float; val x2: Float; val y2: Float
            if (bboxPerAnchor) {
                x1 = bboxData[a * 4 + 0]
                y1 = bboxData[a * 4 + 1]
                x2 = bboxData[a * 4 + 2]
                y2 = bboxData[a * 4 + 3]
            } else {
                x1 = bboxData[0 * numAnchors + a]
                y1 = bboxData[1 * numAnchors + a]
                x2 = bboxData[2 * numAnchors + a]
                y2 = bboxData[3 * numAnchors + a]
            }

            results.add(DetectionResult(
                label = DC_CLASSES[bestClass],
                score = bestScore,
                polygon = listOf(
                    PointF(x1 * scale, y1 * scale),
                    PointF(x2 * scale, y1 * scale),
                    PointF(x2 * scale, y2 * scale),
                    PointF(x1 * scale, y2 * scale)
                ),
                classId = bestClass
            ))
        }

        // NMS
        val sorted = results.sortedByDescending { it.score }
        val kept = mutableListOf<DetectionResult>()
        for (det in sorted) {
            if (kept.size >= MAX_DETECTIONS) break
            val dominated = kept.any { existing ->
                existing.classId == det.classId && polygonIou(existing.polygon, det.polygon) > IOU_THRESHOLD
            }
            if (!dominated) kept.add(det)
        }

        return kept
    }

    // --- YOLO output parsing ---

    private fun parseYoloOutput(outputs: Array<*>): List<DetectionResult> {
        if (outputs.isEmpty()) return emptyList()

        val detTensor = toTensor(outputs[0] ?: return emptyList())
        val data = tensorDataAsFloatArray(detTensor)
        val shape = tensorShape(detTensor)

        val numFeatures = shape[1].toInt()
        val numAnchors = shape[2].toInt()
        val numClasses = DC_CLASSES.size
        val numMaskCoeffs = numFeatures - 4 - numClasses

        val hasMasks = outputs.size > 1 && numMaskCoeffs > 0
        var maskProtos: FloatArray? = null
        var maskH = 0; var maskW = 0
        if (hasMasks) {
            val mt = toTensor(outputs[1] ?: return emptyList())
            maskProtos = tensorDataAsFloatArray(mt)
            val ms = tensorShape(mt)
            maskH = ms[2].toInt(); maskW = ms[3].toInt()
        }

        val cxArr = FloatArray(numAnchors); val cyArr = FloatArray(numAnchors)
        val wArr = FloatArray(numAnchors); val hArr = FloatArray(numAnchors)
        val classArr = IntArray(numAnchors); val scoreArr = FloatArray(numAnchors)
        val coeffArr = Array(numAnchors) { FloatArray(maxOf(numMaskCoeffs, 0)) }
        var count = 0

        for (a in 0 until numAnchors) {
            var bestC = 0; var bestS = 0f
            for (c in 0 until numClasses) {
                val s = data[(4 + c) * numAnchors + a]
                if (s > bestS) { bestS = s; bestC = c }
            }
            if (bestS < CONF_THRESHOLD) continue
            cxArr[count] = data[0 * numAnchors + a]
            cyArr[count] = data[1 * numAnchors + a]
            wArr[count] = data[2 * numAnchors + a]
            hArr[count] = data[3 * numAnchors + a]
            classArr[count] = bestC; scoreArr[count] = bestS
            for (m in 0 until maxOf(numMaskCoeffs, 0))
                coeffArr[count][m] = data[(4 + numClasses + m) * numAnchors + a]
            count++
        }

        val indices = (0 until count).sortedByDescending { scoreArr[it] }
        val kept = mutableListOf<Int>()
        for (idx in indices) {
            if (kept.size >= MAX_DETECTIONS) break
            if (kept.none { k -> classArr[k] == classArr[idx] && boxIou(
                    cxArr[k], cyArr[k], wArr[k], hArr[k],
                    cxArr[idx], cyArr[idx], wArr[idx], hArr[idx]) > IOU_THRESHOLD })
                kept.add(idx)
        }

        val scale = 1.0f / INPUT_SIZE
        return kept.map { i ->
            val polygon = if (hasMasks && maskProtos != null) {
                maskToPolygon(coeffArr[i], maskProtos, maskH, maskW,
                    cxArr[i], cyArr[i], wArr[i], hArr[i], scale)
            } else {
                bboxPolygon(cxArr[i], cyArr[i], wArr[i], hArr[i], scale)
            }
            DetectionResult(DC_CLASSES[classArr[i]], scoreArr[i], polygon, classArr[i])
        }
    }

    // --- Utility functions ---

    private fun sigmoid(x: Float): Float = 1.0f / (1.0f + exp(-x))

    private fun polygonIou(a: List<PointF>, b: List<PointF>): Float {
        // Approximate IoU using bounding box of polygons
        val ax1 = a.minOf { it.x }; val ay1 = a.minOf { it.y }
        val ax2 = a.maxOf { it.x }; val ay2 = a.maxOf { it.y }
        val bx1 = b.minOf { it.x }; val by1 = b.minOf { it.y }
        val bx2 = b.maxOf { it.x }; val by2 = b.maxOf { it.y }
        val ix = maxOf(0f, minOf(ax2, bx2) - maxOf(ax1, bx1))
        val iy = maxOf(0f, minOf(ay2, by2) - maxOf(ay1, by1))
        val inter = ix * iy
        val areaA = (ax2 - ax1) * (ay2 - ay1)
        val areaB = (bx2 - bx1) * (by2 - by1)
        return inter / (areaA + areaB - inter + 1e-6f)
    }

    private fun bboxPolygon(cx: Float, cy: Float, w: Float, h: Float, s: Float): List<PointF> {
        val x1 = (cx - w / 2) * s; val y1 = (cy - h / 2) * s
        val x2 = (cx + w / 2) * s; val y2 = (cy + h / 2) * s
        return listOf(PointF(x1, y1), PointF(x2, y1), PointF(x2, y2), PointF(x1, y2))
    }

    private fun maskToPolygon(
        coeffs: FloatArray, protos: FloatArray,
        mH: Int, mW: Int,
        cx: Float, cy: Float, w: Float, h: Float, scale: Float
    ): List<PointF> {
        val cellW = INPUT_SIZE.toFloat() / mW
        val cellH = INPUT_SIZE.toFloat() / mH
        val bx1 = ((cx - w / 2) / cellW).toInt().coerceIn(0, mW - 1)
        val by1 = ((cy - h / 2) / cellH).toInt().coerceIn(0, mH - 1)
        val bx2 = ((cx + w / 2) / cellW).toInt().coerceIn(0, mW - 1)
        val by2 = ((cy + h / 2) / cellH).toInt().coerceIn(0, mH - 1)
        val edge = mutableListOf<PointF>()
        for (y in by1..by2) {
            for (x in bx1..bx2) {
                var v = 0f
                for (c in coeffs.indices) v += coeffs[c] * protos[c * mH * mW + y * mW + x]
                if (v <= 0f) continue
                val onBorder = x == bx1 || x == bx2 || y == by1 || y == by2
                if (onBorder || isEdgePixel(coeffs, protos, mH, mW, x, y))
                    edge.add(PointF(x * cellW * scale, y * cellH * scale))
            }
        }
        if (edge.size < 3) return bboxPolygon(cx, cy, w, h, scale)
        val avgX = edge.map { it.x }.average().toFloat()
        val avgY = edge.map { it.y }.average().toFloat()
        val step = maxOf(1, edge.size / 24)
        return edge.sortedBy { Math.atan2((it.y - avgY).toDouble(), (it.x - avgX).toDouble()) }
            .filterIndexed { i, _ -> i % step == 0 }
    }

    private fun isEdgePixel(coeffs: FloatArray, protos: FloatArray, mH: Int, mW: Int, x: Int, y: Int): Boolean {
        if (x <= 0 || x >= mW - 1) return true
        for (dx in intArrayOf(-1, 1)) {
            var v = 0f
            for (c in coeffs.indices) v += coeffs[c] * protos[c * mH * mW + y * mW + x + dx]
            if (v <= 0f) return true
        }
        return false
    }

    private fun boxIou(cx1: Float, cy1: Float, w1: Float, h1: Float,
                       cx2: Float, cy2: Float, w2: Float, h2: Float): Float {
        val ax1 = cx1 - w1/2; val ay1 = cy1 - h1/2; val ax2 = cx1 + w1/2; val ay2 = cy1 + h1/2
        val bx1 = cx2 - w2/2; val by1 = cy2 - h2/2; val bx2 = cx2 + w2/2; val by2 = cy2 + h2/2
        val ix = maxOf(0f, minOf(ax2, bx2) - maxOf(ax1, bx1))
        val iy = maxOf(0f, minOf(ay2, by2) - maxOf(ay1, by1))
        val inter = ix * iy
        return inter / (w1*h1 + w2*h2 - inter + 1e-6f)
    }

    // --- Image conversion ---

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer
        val nv21 = ByteArray(yBuffer.remaining() + uBuffer.remaining() + vBuffer.remaining())
        yBuffer.get(nv21, 0, yBuffer.remaining())
        vBuffer.get(nv21, yBuffer.capacity(), vBuffer.remaining())
        uBuffer.get(nv21, yBuffer.capacity() + vBuffer.capacity(), uBuffer.remaining())
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 90, out)
        return BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath
        context.assets.open(assetName).use { input ->
            FileOutputStream(file).use { output -> input.copyTo(output) }
        }
        return file.absolutePath
    }

    fun shutdown() {
        module?.let { m ->
            m.javaClass.methods.firstOrNull { it.name == "destroy" && it.parameterTypes.isEmpty() }?.invoke(m)
        }
        module = null
        isReady = false
    }

    // --- ExecuTorch reflection helpers ---

    private fun loadExecuTorchModule(modelPath: String): Any {
        val moduleClass = Class.forName("org.pytorch.executorch.Module")
        val loadMethod = moduleClass.getMethod("load", String::class.java)
        return requireNotNull(loadMethod.invoke(null, modelPath)) { "Module.load returned null" }
    }

    private fun createTensorFromBlob(buffer: FloatBuffer, shape: LongArray): Any {
        val tensorClass = Class.forName("org.pytorch.executorch.Tensor")
        val m = tensorClass.getMethod("fromBlob", FloatBuffer::class.java, LongArray::class.java)
        return requireNotNull(m.invoke(null, buffer, shape)) { "Tensor.fromBlob returned null" }
    }

    private fun wrapTensorInEValue(tensor: Any): Any {
        val eValueClass = Class.forName("org.pytorch.executorch.EValue")
        val m = eValueClass.methods.firstOrNull { it.name == "from" && it.parameterTypes.size == 1 }
            ?: error("EValue.from not found")
        return requireNotNull(m.invoke(null, tensor)) { "EValue.from returned null" }
    }

    private fun runModuleForward(loadedModule: Any, input: Any): Array<*> {
        val m = loadedModule.javaClass.methods.firstOrNull { it.name == "forward" && it.parameterTypes.size == 1 }
            ?: error("Module.forward not found")
        return m.invoke(loadedModule, input) as? Array<*> ?: error("forward returned unexpected type")
    }

    private fun toTensor(value: Any): Any {
        val m = value.javaClass.methods.firstOrNull { it.name == "toTensor" && it.parameterTypes.isEmpty() }
            ?: error("EValue.toTensor not found")
        return requireNotNull(m.invoke(value)) { "toTensor returned null" }
    }

    private fun tensorDataAsFloatArray(tensor: Any): FloatArray {
        val m = tensor.javaClass.methods.firstOrNull { it.name == "getDataAsFloatArray" && it.parameterTypes.isEmpty() }
            ?: error("Tensor.getDataAsFloatArray not found")
        return m.invoke(tensor) as? FloatArray ?: error("data was not FloatArray")
    }

    private fun tensorShape(tensor: Any): LongArray {
        val m = tensor.javaClass.methods.firstOrNull { it.name == "shape" && it.parameterTypes.isEmpty() }
            ?: tensor.javaClass.methods.firstOrNull { it.name == "getShape" && it.parameterTypes.isEmpty() }
            ?: error("Tensor shape accessor not found")
        return m.invoke(tensor) as? LongArray ?: error("shape was not LongArray")
    }
}

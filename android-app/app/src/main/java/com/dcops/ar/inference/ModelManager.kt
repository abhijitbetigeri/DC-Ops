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

class ModelManager {

    companion object {
        const val MODEL_FILENAME = "dc_ops_yolov8n_seg.pte"
        const val QNN_MODEL_FILENAME = "dc_ops_retinanet_qnn.pte"
        const val INPUT_SIZE = 640
        const val CONF_THRESHOLD = 0.35f
        const val IOU_THRESHOLD = 0.45f
        const val MAX_DETECTIONS = 50

        val DC_CLASSES = arrayOf(
            "server rack", "compute tray", "NVLink switch tray", "network switch",
            "power shelf", "cable", "network port", "LED indicator",
            "label", "fan", "cooling manifold", "cable cartridge",
            "power connector", "drive bay", "management port", "DPU"
        )
    }

    private var module: Any? = null
    private var isReady = false
    private val inputBuffer = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)

    fun init(context: Context, onReady: (Boolean) -> Unit) {
        Thread {
            try {
                val modelPath = assetFilePath(context, MODEL_FILENAME)
                module = loadExecuTorchModule(modelPath)
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
        val loadedModule = module ?: run {
            imageProxy.close()
            return
        }
        if (!isReady) {
            imageProxy.close()
            return
        }

        try {
            val bitmap = imageProxyToBitmap(imageProxy)
            val scaled = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
            val inputTensor = bitmapToEValue(scaled)

            val outputs = runModuleForward(loadedModule, inputTensor)
            val results = parseOutput(outputs)
            onResult(results)
        } catch (e: Exception) {
            e.printStackTrace()
            onResult(emptyList())
        } finally {
            imageProxy.close()
        }
    }

    private fun bitmapToEValue(bitmap: Bitmap): Any {
        inputBuffer.rewind()
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) inputBuffer.put(((pixel shr 16) and 0xFF) / 255.0f)
        for (pixel in pixels) inputBuffer.put(((pixel shr 8) and 0xFF) / 255.0f)
        for (pixel in pixels) inputBuffer.put((pixel and 0xFF) / 255.0f)

        inputBuffer.rewind()
        val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        val tensor = createTensorFromBlob(inputBuffer, shape)
        return wrapTensorInEValue(tensor)
    }

    private fun parseOutput(outputs: Array<*>): List<DetectionResult> {
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
        var maskH = 0
        var maskW = 0
        if (hasMasks) {
            val mt = toTensor(outputs[1] ?: return emptyList())
            maskProtos = tensorDataAsFloatArray(mt)
            val maskShape = tensorShape(mt)
            maskH = maskShape[2].toInt()
            maskW = maskShape[3].toInt()
        }

        // Collect raw detections above confidence threshold
        val cxArr = FloatArray(numAnchors)
        val cyArr = FloatArray(numAnchors)
        val wArr = FloatArray(numAnchors)
        val hArr = FloatArray(numAnchors)
        val classArr = IntArray(numAnchors)
        val scoreArr = FloatArray(numAnchors)
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
            classArr[count] = bestC
            scoreArr[count] = bestS
            for (m in 0 until maxOf(numMaskCoeffs, 0)) {
                coeffArr[count][m] = data[(4 + numClasses + m) * numAnchors + a]
            }
            count++
        }

        // Sort by score descending
        val indices = (0 until count).sortedByDescending { scoreArr[it] }

        // NMS
        val kept = mutableListOf<Int>()
        for (idx in indices) {
            if (kept.size >= MAX_DETECTIONS) break
            val dominated = kept.any { k ->
                classArr[k] == classArr[idx] && boxIou(
                    cxArr[k], cyArr[k], wArr[k], hArr[k],
                    cxArr[idx], cyArr[idx], wArr[idx], hArr[idx]
                ) > IOU_THRESHOLD
            }
            if (!dominated) kept.add(idx)
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
                if (onBorder || isEdgePixel(coeffs, protos, mH, mW, x, y)) {
                    edge.add(PointF(x * cellW * scale, y * cellH * scale))
                }
            }
        }

        if (edge.size < 3) return bboxPolygon(cx, cy, w, h, scale)

        val avgX = edge.map { it.x }.average().toFloat()
        val avgY = edge.map { it.y }.average().toFloat()
        val step = maxOf(1, edge.size / 24)
        return edge
            .sortedBy { Math.atan2((it.y - avgY).toDouble(), (it.x - avgX).toDouble()) }
            .filterIndexed { i, _ -> i % step == 0 }
    }

    private fun isEdgePixel(
        coeffs: FloatArray, protos: FloatArray, mH: Int, mW: Int, x: Int, y: Int
    ): Boolean {
        if (x <= 0 || x >= mW - 1) return true
        for (dx in intArrayOf(-1, 1)) {
            var v = 0f
            for (c in coeffs.indices) v += coeffs[c] * protos[c * mH * mW + y * mW + x + dx]
            if (v <= 0f) return true
        }
        return false
    }

    private fun boxIou(
        cx1: Float, cy1: Float, w1: Float, h1: Float,
        cx2: Float, cy2: Float, w2: Float, h2: Float
    ): Float {
        val ax1 = cx1 - w1 / 2; val ay1 = cy1 - h1 / 2
        val ax2 = cx1 + w1 / 2; val ay2 = cy1 + h1 / 2
        val bx1 = cx2 - w2 / 2; val by1 = cy2 - h2 / 2
        val bx2 = cx2 + w2 / 2; val by2 = cy2 + h2 / 2
        val ix = maxOf(0f, minOf(ax2, bx2) - maxOf(ax1, bx1))
        val iy = maxOf(0f, minOf(ay2, by2) - maxOf(ay1, by1))
        val inter = ix * iy
        return inter / (w1 * h1 + w2 * h2 - inter + 1e-6f)
    }

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

    fun switchModel(context: Context, modelFile: String, onReady: (Boolean) -> Unit) {
        isReady = false
        Thread {
            try {
                module?.destroy()
                val modelPath = assetFilePath(context, modelFile)
                module = Module.load(modelPath)
                isReady = true
                onReady(true)
            } catch (e: Exception) {
                e.printStackTrace()
                isReady = false
                onReady(false)
            }
        }.start()
    }

    fun shutdown() {
        module?.let { loadedModule ->
            val destroyMethod = loadedModule.javaClass.methods.firstOrNull {
                it.name == "destroy" && it.parameterTypes.isEmpty()
            }
            destroyMethod?.invoke(loadedModule)
        }
        module = null
        isReady = false
    }

    private fun loadExecuTorchModule(modelPath: String): Any {
        val moduleClass = Class.forName("org.pytorch.executorch.Module")
        val loadMethod = moduleClass.getMethod("load", String::class.java)
        return requireNotNull(loadMethod.invoke(null, modelPath)) {
            "ExecuTorch Module.load returned null"
        }
    }

    private fun createTensorFromBlob(buffer: FloatBuffer, shape: LongArray): Any {
        val tensorClass = Class.forName("org.pytorch.executorch.Tensor")
        val fromBlobMethod = tensorClass.getMethod("fromBlob", FloatBuffer::class.java, LongArray::class.java)
        return requireNotNull(fromBlobMethod.invoke(null, buffer, shape)) {
            "ExecuTorch Tensor.fromBlob returned null"
        }
    }

    private fun wrapTensorInEValue(tensor: Any): Any {
        val eValueClass = Class.forName("org.pytorch.executorch.EValue")
        val fromMethod = eValueClass.methods.firstOrNull {
            it.name == "from" && it.parameterTypes.size == 1
        } ?: error("ExecuTorch EValue.from(Tensor) not found")
        return requireNotNull(fromMethod.invoke(null, tensor)) {
            "ExecuTorch EValue.from returned null"
        }
    }

    private fun runModuleForward(loadedModule: Any, input: Any): Array<*> {
        val forwardMethod = loadedModule.javaClass.methods.firstOrNull {
            it.name == "forward" && it.parameterTypes.size == 1
        } ?: error("ExecuTorch Module.forward(EValue) not found")
        val output = forwardMethod.invoke(loadedModule, input)
        return output as? Array<*> ?: error("ExecuTorch Module.forward returned unexpected value")
    }

    private fun toTensor(value: Any): Any {
        val toTensorMethod = value.javaClass.methods.firstOrNull {
            it.name == "toTensor" && it.parameterTypes.isEmpty()
        } ?: error("ExecuTorch EValue.toTensor() not found")
        return requireNotNull(toTensorMethod.invoke(value)) {
            "ExecuTorch EValue.toTensor returned null"
        }
    }

    private fun tensorDataAsFloatArray(tensor: Any): FloatArray {
        val dataMethod = tensor.javaClass.methods.firstOrNull {
            it.name == "getDataAsFloatArray" && it.parameterTypes.isEmpty()
        } ?: error("ExecuTorch Tensor.getDataAsFloatArray() not found")
        return dataMethod.invoke(tensor) as? FloatArray
            ?: error("ExecuTorch Tensor data was not FloatArray")
    }

    private fun tensorShape(tensor: Any): LongArray {
        val shapeMethod = tensor.javaClass.methods.firstOrNull {
            it.name == "shape" && it.parameterTypes.isEmpty()
        } ?: tensor.javaClass.methods.firstOrNull {
            it.name == "getShape" && it.parameterTypes.isEmpty()
        } ?: error("ExecuTorch Tensor shape accessor not found")
        return shapeMethod.invoke(tensor) as? LongArray
            ?: error("ExecuTorch Tensor shape was not LongArray")
    }
}

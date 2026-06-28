package com.dcops.ar.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

/**
 * QNN / NPU variant.
 *
 * The .pte is the YOLOv8n-seg **backbone** lowered to the Qualcomm QNN (HTP) backend.
 * It runs on the Hexagon NPU and returns the RAW, pre-decode head outputs:
 *   [0] (1,80,80,80)  per-scale box(64 DFL)+cls(16), stride 8
 *   [1] (1,80,40,40)  stride 16
 *   [2] (1,80,20,20)  stride 32
 *   [3] (1,32,8400)   mask coefficients (anchor order = scale0,scale1,scale2 row-major)
 *   [4] (1,32,160,160) mask prototypes
 *
 * The detection decode (DFL -> distances, anchors, dist2bbox, class sigmoid, mask
 * assembly, NMS) is done here in Kotlin -- those ops don't lower to QNN, so they're
 * kept off the NPU. This is the standard "delegate backbone, decode on host" pattern.
 */
class ModelManager {

    companion object {
        const val MODEL_FILENAME = "dc_ops_yolov8n_seg_backbone_qnn.pte"
        const val INPUT_SIZE = 640
        const val REG_MAX = 16
        const val NUM_CLASSES = 16
        const val NUM_MASK = 32
        const val CONF_THRESHOLD = 0.35f
        const val IOU_THRESHOLD = 0.45f
        const val MAX_DETECTIONS = 50

        val DC_CLASSES = arrayOf(
            "server rack", "compute tray", "NVLink switch tray", "network switch",
            "power shelf", "cable", "network port", "LED indicator",
            "label", "fan", "cooling manifold", "cable cartridge",
            "power connector", "drive bay", "management port", "DPU"
        )

        // (grid, stride): 640/8=80, 640/16=40, 640/32=20
        private val SCALES = arrayOf(intArrayOf(80, 8), intArrayOf(40, 16), intArrayOf(20, 32))
    }

    private var module: Module? = null
    private var isReady = false
    private val inputBuffer = ByteBuffer
        .allocateDirect(3 * INPUT_SIZE * INPUT_SIZE * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()

    fun init(context: Context, onReady: (Boolean) -> Unit) {
        Thread {
            try {
                // The QNN HTP backend dlopens libQnnHtp* and loads the v79 skel onto the
                // Hexagon DSP via fastRPC, which searches ADSP_LIBRARY_PATH / LD_LIBRARY_PATH.
                // Point them at the app's packaged jniLibs dir or QnnManager init crashes.
                val nativeDir = context.applicationInfo.nativeLibraryDir
                try {
                    android.system.Os.setenv("ADSP_LIBRARY_PATH", nativeDir, true)
                    android.system.Os.setenv("LD_LIBRARY_PATH", nativeDir, true)
                } catch (_: Exception) {}
                module = Module.load(assetFilePath(context, MODEL_FILENAME))
                isReady = true
                onReady(true)
            } catch (e: Exception) {
                e.printStackTrace(); isReady = false; onReady(false)
            }
        }.start()
    }

    fun processFrame(imageProxy: ImageProxy, onResult: (List<DetectionResult>) -> Unit) {
        if (!isReady || module == null) { imageProxy.close(); return }
        try {
            val bmp = imageProxyToBitmap(imageProxy)
            val scaled = Bitmap.createScaledBitmap(bmp, INPUT_SIZE, INPUT_SIZE, true)
            val outputs = module!!.forward(bitmapToTensor(scaled))
            val results = decode(outputs)
            imageProxy.close()
            onResult(results)
        } catch (e: Exception) {
            e.printStackTrace(); imageProxy.close(); onResult(emptyList())
        }
    }

    /** Debug: run inference on a bundled asset image through the same path. */
    fun processTestAsset(context: Context, assetName: String, onResult: (List<DetectionResult>) -> Unit) {
        if (!isReady || module == null) { onResult(emptyList()); return }
        Thread {
            try {
                val bmp = context.assets.open(assetName).use { BitmapFactory.decodeStream(it) }
                val scaled = Bitmap.createScaledBitmap(bmp, INPUT_SIZE, INPUT_SIZE, true)
                val outputs = module!!.forward(bitmapToTensor(scaled))
                onResult(decode(outputs))
            } catch (e: Exception) {
                e.printStackTrace(); onResult(emptyList())
            }
        }.start()
    }

    private fun bitmapToTensor(bitmap: Bitmap): EValue {
        inputBuffer.rewind()
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (p in pixels) inputBuffer.put(((p shr 16) and 0xFF) / 255.0f)
        for (p in pixels) inputBuffer.put(((p shr 8) and 0xFF) / 255.0f)
        for (p in pixels) inputBuffer.put((p and 0xFF) / 255.0f)
        inputBuffer.rewind()
        return EValue.from(Tensor.fromBlob(inputBuffer, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())))
    }

    private class Det(
        val cx: Float, val cy: Float, val w: Float, val h: Float,
        val cls: Int, val score: Float, val coeff: FloatArray
    )

    private fun sigmoid(x: Float) = 1f / (1f + exp(-x))

    /** Decode raw NPU outputs -> detections (DFL + anchors + dist2bbox + sigmoid + NMS + masks). */
    private fun decode(outputs: Array<EValue>): List<DetectionResult> {
        if (outputs.size < 5) return emptyList()
        // Identify outputs by SHAPE (QNN lowering may reorder them), not by fixed index:
        //   per-scale box+cls : (1,80,H,W)   mask coeff : (1,32,8400)   proto : (1,32,160,160)
        val tensors = outputs.map { it.toTensor() }
        android.util.Log.i("DCOPS", "qnn outputs: " + tensors.joinToString { t -> t.shape().joinToString("x") })
        var mcT: Tensor? = null; var protoT: Tensor? = null
        val perScale = ArrayList<Tensor>()
        for (t in tensors) {
            val s = t.shape()
            when {
                s.size == 4 && s[1].toInt() == 32 -> protoT = t
                s.size == 3 && s[1].toInt() == 32 -> mcT = t
                s.size == 4 && s[1].toInt() == 80 -> perScale.add(t)
            }
        }
        if (mcT == null || protoT == null || perScale.size < 3) {
            android.util.Log.w("DCOPS", "unexpected output shapes; cannot map")
            return emptyList()
        }
        perScale.sortByDescending { it.shape()[2].toInt() }      // grids 80,40,20
        val mc = mcT.dataAsFloatArray
        val numAnchors = mcT.shape()[2].toInt()                  // 8400
        val proto = protoT.dataAsFloatArray
        val maskH = protoT.shape()[2].toInt(); val maskW = protoT.shape()[3].toInt()

        val dets = ArrayList<Det>()
        var anchorOffset = 0
        for (s in 0 until 3) {
            val grid = SCALES[s][0]; val stride = SCALES[s][1].toFloat()
            val hw = grid * grid
            val d = perScale[s].dataAsFloatArray   // (1,80,grid,grid) NCHW
            for (yy in 0 until grid) for (xx in 0 until grid) {
                val cell = yy * grid + xx
                // best class (sigmoid of channels 64..79)
                var bestC = 0; var bestLogit = -1e30f
                for (c in 0 until NUM_CLASSES) {
                    val v = d[(4 * REG_MAX + c) * hw + cell]
                    if (v > bestLogit) { bestLogit = v; bestC = c }
                }
                val score = sigmoid(bestLogit)
                if (score < CONF_THRESHOLD) continue
                // DFL: 4 sides, softmax over REG_MAX bins, expected value
                val dist = FloatArray(4)
                for (k in 0 until 4) {
                    var mx = -1e30f
                    for (b in 0 until REG_MAX) { val v = d[(k * REG_MAX + b) * hw + cell]; if (v > mx) mx = v }
                    var sum = 0f; val e = FloatArray(REG_MAX)
                    for (b in 0 until REG_MAX) { val ev = exp(d[(k * REG_MAX + b) * hw + cell] - mx); e[b] = ev; sum += ev }
                    var acc = 0f
                    for (b in 0 until REG_MAX) acc += b * (e[b] / sum)
                    dist[k] = acc
                }
                val ax = xx + 0.5f; val ay = yy + 0.5f
                val x1 = (ax - dist[0]) * stride; val y1 = (ay - dist[1]) * stride
                val x2 = (ax + dist[2]) * stride; val y2 = (ay + dist[3]) * stride
                val anchorIdx = anchorOffset + cell
                val coeff = FloatArray(NUM_MASK) { mc[it * numAnchors + anchorIdx] }
                dets.add(Det((x1 + x2) / 2f, (y1 + y2) / 2f, x2 - x1, y2 - y1, bestC, score, coeff))
            }
            anchorOffset += hw
        }

        // NMS (per class)
        dets.sortByDescending { it.score }
        val kept = ArrayList<Det>()
        for (cand in dets) {
            if (kept.size >= MAX_DETECTIONS) break
            if (kept.any { it.cls == cand.cls && iou(it, cand) > IOU_THRESHOLD }) continue
            kept.add(cand)
        }

        val scale = 1.0f / INPUT_SIZE
        return kept.map {
            val poly = maskToPolygon(it.coeff, proto, maskH, maskW, it.cx, it.cy, it.w, it.h, scale)
            DetectionResult(DC_CLASSES[it.cls], it.score, poly, it.cls)
        }
    }

    private fun iou(a: Det, b: Det): Float {
        val ax1 = a.cx - a.w / 2; val ay1 = a.cy - a.h / 2; val ax2 = a.cx + a.w / 2; val ay2 = a.cy + a.h / 2
        val bx1 = b.cx - b.w / 2; val by1 = b.cy - b.h / 2; val bx2 = b.cx + b.w / 2; val by2 = b.cy + b.h / 2
        val ix = maxOf(0f, minOf(ax2, bx2) - maxOf(ax1, bx1))
        val iy = maxOf(0f, minOf(ay2, by2) - maxOf(ay1, by1))
        val inter = ix * iy
        return inter / (a.w * a.h + b.w * b.h - inter + 1e-6f)
    }

    private fun bboxPolygon(cx: Float, cy: Float, w: Float, h: Float, s: Float): List<PointF> {
        val x1 = (cx - w / 2) * s; val y1 = (cy - h / 2) * s
        val x2 = (cx + w / 2) * s; val y2 = (cy + h / 2) * s
        return listOf(PointF(x1, y1), PointF(x2, y1), PointF(x2, y2), PointF(x1, y2))
    }

    private fun maskToPolygon(
        coeffs: FloatArray, protos: FloatArray, mH: Int, mW: Int,
        cx: Float, cy: Float, w: Float, h: Float, scale: Float
    ): List<PointF> {
        val cellW = INPUT_SIZE.toFloat() / mW; val cellH = INPUT_SIZE.toFloat() / mH
        val bx1 = ((cx - w / 2) / cellW).toInt().coerceIn(0, mW - 1)
        val by1 = ((cy - h / 2) / cellH).toInt().coerceIn(0, mH - 1)
        val bx2 = ((cx + w / 2) / cellW).toInt().coerceIn(0, mW - 1)
        val by2 = ((cy + h / 2) / cellH).toInt().coerceIn(0, mH - 1)
        val edge = mutableListOf<PointF>()
        for (y in by1..by2) for (x in bx1..bx2) {
            var v = 0f
            for (c in coeffs.indices) v += coeffs[c] * protos[c * mH * mW + y * mW + x]
            if (v <= 0f) continue
            val onBorder = x == bx1 || x == bx2 || y == by1 || y == by2
            if (onBorder || isEdgePixel(coeffs, protos, mH, mW, x, y))
                edge.add(PointF(x * cellW * scale, y * cellH * scale))
        }
        if (edge.size < 3) return bboxPolygon(cx, cy, w, h, scale)
        val avgX = edge.map { it.x }.average().toFloat(); val avgY = edge.map { it.y }.average().toFloat()
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
        context.assets.open(assetName).use { input -> FileOutputStream(file).use { input.copyTo(it) } }
        return file.absolutePath
    }

    fun shutdown() { module?.destroy(); module = null; isReady = false }
}

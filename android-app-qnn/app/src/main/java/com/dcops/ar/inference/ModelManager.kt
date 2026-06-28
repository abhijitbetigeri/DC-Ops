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
import java.io.DataInputStream
import java.io.DataOutputStream
import java.net.InetSocketAddress
import java.net.Socket
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * QNN / NPU variant — **NPU-over-shell-bridge, server-side decode**.
 *
 * Inference runs on the Hexagon v79 NPU inside a persistent shell-context server
 * (`qnn_socket_server`, /data/local/tmp) because a retail S25 won't let a normal app
 * open an unsigned-PD cDSP session. As of the real-time pass, the server also does the
 * full YOLOv8-seg decode (DFL, dist2bbox, sigmoid, NMS, mask->polygon) in C++ and returns
 * only the final detections, so this app just sends a frame and renders polygons.
 *
 * Wire protocol (127.0.0.1:8765):
 *   app -> server:  int32 ctrl (LE; server floor in 1/1000ths 0..1000, or -1
 *                   for the server default CONF_FLOOR), then
 *                   RGBA8888 bytes (640*640*4) from Bitmap.copyPixelsToBuffer
 *   server -> app:  int32 payload_len (LE), then payload (LE):
 *                     int32 num_dets
 *                     repeat: int32 class_id, float score, int32 n_pts,
 *                             repeat n_pts: float x, float y   (normalized 0..1)
 * The server emits all detections >= 0.20; this app filters by [confThreshold] (the slider).
 */
class ModelManager {

    companion object {
        const val INPUT_SIZE = 640                               // fallback for old servers w/o handshake
        const val IN_BYTES = INPUT_SIZE * INPUT_SIZE * 4          // RGBA8888, 1,638,400

        const val HANDSHAKE_MAGIC = 0x4D4F444C                   // 'MODL' (LE int32)

        // w8a16 quantization compresses confidence (peak ~0.44 on this model); 0.30 surfaces
        // the genuine detections. The slider tunes this live.
        const val DEFAULT_CONF = 0.20f

        // Default server-side conf floor. The "Server floor" slider tunes this live;
        // drag down to see more, up to be stricter. Lower than the C++ default so the
        // (lossy) live screen-pointing demo still surfaces real detections.
        const val DEFAULT_SERVER_FLOOR = 0.10f

        const val SERVER_HOST = "127.0.0.1"
        const val SERVER_PORT = 8765

        val DC_CLASSES = arrayOf(
            "server rack", "compute tray", "NVLink switch tray", "network switch",
            "power shelf", "cable", "network port", "LED indicator",
            "label", "fan", "cooling manifold", "cable cartridge",
            "power connector", "drive bay", "management port", "DPU"
        )
    }

    /** Min detection confidence to display (0..1). Tunable live from the UI slider. */
    @Volatile var confThreshold = DEFAULT_CONF

    /**
     * Server-side conf floor (0..1), sent in the per-frame control word so the NPU
     * server only emits detections at/above it. Tunable live from the "Server floor"
     * slider; defaults to [DEFAULT_SERVER_FLOOR].
     */
    @Volatile var serverFloor = DEFAULT_SERVER_FLOOR

    @Volatile private var isReady = false
    private var socket: Socket? = null
    private var sockIn: DataInputStream? = null
    private var sockOut: DataOutputStream? = null
    private val ioLock = Any()

    // Learned from the server handshake (fall back to the compiled defaults for an old server).
    @Volatile private var inputSize = INPUT_SIZE
    @Volatile private var inBytes = IN_BYTES
    @Volatile private var labels: List<String> = DC_CLASSES.toList()

    // (Re)allocated on handshake whenever inputSize changes; sized to the current inBytes.
    private var inBuf = ByteBuffer.allocateDirect(IN_BYTES).order(ByteOrder.LITTLE_ENDIAN)
    private var inHeap = ByteArray(IN_BYTES)
    private val ctrlHeader = ByteArray(4)        // per-frame LE int32 control word (bit0 = disable conf floor)
    private var respBuf = ByteArray(64 * 1024)   // grows if a frame ever needs more
    private var frameCounter = 0

    fun init(context: Context, onReady: (Boolean) -> Unit) {
        Thread {
            val ok = synchronized(ioLock) { connectLocked() }
            isReady = true
            if (!ok) android.util.Log.w("DCOPS", "NPU server not reachable at $SERVER_HOST:$SERVER_PORT yet")
            onReady(true)
        }.start()
    }

    /** Establish a fresh connection. Caller MUST hold ioLock. */
    private fun connectLocked(): Boolean {
        closeSocketLocked()
        return try {
            val s = Socket(); s.tcpNoDelay = true
            s.connect(InetSocketAddress(SERVER_HOST, SERVER_PORT), 2000)
            socket = s
            sockOut = DataOutputStream(s.getOutputStream())
            sockIn = DataInputStream(s.getInputStream())
            android.util.Log.i("DCOPS", "connected to NPU server $SERVER_HOST:$SERVER_PORT")
            readHandshakeLocked(sockIn!!)
            true
        } catch (e: Exception) {
            android.util.Log.w("DCOPS", "NPU connect failed: ${e.message}")
            closeSocketLocked()
            false
        }
    }

    /**
     * Read the server's startup handshake BEFORE any frame is sent (caller holds ioLock,
     * called from connectLocked so it completes on the connect path):
     *   int32 magic, int32 input_w, int32 input_h, int32 num_classes,
     *   repeat num_classes: int32 byte_len, byte_len UTF-8 bytes (the label).
     * On magic mismatch or any read failure we fall back to the compiled defaults
     * (640 + DC_CLASSES) so the app still works against an old, handshake-less server.
     */
    private fun readHandshakeLocked(din: DataInputStream) {
        try {
            val magic = readIntLE(din)
            if (magic != HANDSHAKE_MAGIC) {
                android.util.Log.w("DCOPS",
                    "handshake: bad magic 0x%08X, using defaults (input=$INPUT_SIZE)".format(magic))
                applySpec(INPUT_SIZE, INPUT_SIZE, DC_CLASSES.toList())
                return
            }
            val w = readIntLE(din)
            val h = readIntLE(din)
            val numClasses = readIntLE(din)
            val lbls = ArrayList<String>(if (numClasses in 0..4096) numClasses else 0)
            for (i in 0 until numClasses) {
                val byteLen = readIntLE(din)
                val buf = ByteArray(byteLen)
                din.readFully(buf, 0, byteLen)
                lbls.add(String(buf, Charsets.UTF_8))
            }
            applySpec(w, h, lbls)
            android.util.Log.i("DCOPS", "handshake: input=$inputSize classes=${labels.size} $labels")
        } catch (e: Exception) {
            android.util.Log.w("DCOPS", "handshake read failed (${e.message}); using defaults")
            applySpec(INPUT_SIZE, INPUT_SIZE, DC_CLASSES.toList())
        }
    }

    /** Apply a learned (or fallback) spec: set inputSize/inBytes/labels and (re)allocate buffers. */
    private fun applySpec(w: Int, h: Int, lbls: List<String>) {
        val size = if (w > 0) w else INPUT_SIZE   // square model; w==h expected
        labels = if (lbls.isNotEmpty()) lbls else DC_CLASSES.toList()
        if (size != inputSize || inHeap.size != size * size * 4) {
            inputSize = size
            inBytes = size * size * 4
            inBuf = ByteBuffer.allocateDirect(inBytes).order(ByteOrder.LITTLE_ENDIAN)
            inHeap = ByteArray(inBytes)
        }
    }

    /** Read a little-endian int32. */
    private fun readIntLE(din: DataInputStream): Int {
        val b0 = din.readUnsignedByte(); val b1 = din.readUnsignedByte()
        val b2 = din.readUnsignedByte(); val b3 = din.readUnsignedByte()
        return b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
    }

    private fun closeSocketLocked() {
        try { socket?.close() } catch (_: Exception) {}
        socket = null; sockIn = null; sockOut = null
    }

    /** Send one frame to the NPU server and read back the decoded detections. */
    private fun infer(bitmap: Bitmap): List<DetectionResult> {
        synchronized(ioLock) {
            val tA = System.nanoTime()
            // Preprocess: copy raw RGBA8888 bytes in one native memcpy (no per-pixel Kotlin loop).
            inBuf.clear()
            bitmap.copyPixelsToBuffer(inBuf)
            inBuf.rewind(); inBuf.get(inHeap)
            val tB = System.nanoTime()

            for (attempt in 0..1) {
                if (sockOut == null || sockIn == null) {
                    if (!connectLocked()) { Thread.sleep(300); continue }
                }
                try {
                    // Per-frame control word (LE int32): server floor in 1/1000ths (0..1000).
                    val ctrl = (serverFloor * 1000f).toInt().coerceIn(0, 1000)
                    ctrlHeader[0] = (ctrl and 0xFF).toByte()
                    ctrlHeader[1] = ((ctrl shr 8) and 0xFF).toByte()
                    ctrlHeader[2] = ((ctrl shr 16) and 0xFF).toByte()
                    ctrlHeader[3] = ((ctrl shr 24) and 0xFF).toByte()
                    sockOut!!.write(ctrlHeader, 0, 4)
                    sockOut!!.write(inHeap, 0, inBytes)
                    sockOut!!.flush()
                    val len = readLenLE(sockIn!!)
                    if (respBuf.size < len) respBuf = ByteArray(len)
                    sockIn!!.readFully(respBuf, 0, len)
                    val tC = System.nanoTime()
                    val out = parseDetections(respBuf, len)
                    val tD = System.nanoTime()
                    if (frameCounter++ % 15 == 0) android.util.Log.i("DCOPS",
                        "prep=%.1f net=%.1f parse=%.1f ms (dets=%d, %d B)".format(
                            (tB - tA) / 1e6, (tC - tB) / 1e6, (tD - tC) / 1e6, out.size, len + 4))
                    return out
                } catch (e: Exception) {
                    android.util.Log.w("DCOPS", "NPU infer attempt $attempt failed: ${e.message}")
                    closeSocketLocked()
                }
            }
        }
        return emptyList()
    }

    /** Read a little-endian int32 length prefix. */
    private fun readLenLE(din: DataInputStream): Int = readIntLE(din)

    /** Parse the server's detection payload, filtering by the live confidence threshold. */
    private fun parseDetections(bytes: ByteArray, len: Int): List<DetectionResult> {
        val bb = ByteBuffer.wrap(bytes, 0, len).order(ByteOrder.LITTLE_ENDIAN)
        val n = bb.int
        val results = ArrayList<DetectionResult>(n)
        for (i in 0 until n) {
            val cls = bb.int
            val score = bb.float
            val nPts = bb.int
            // Always consume the points so the buffer position stays aligned.
            val poly = ArrayList<PointF>(nPts)
            for (p in 0 until nPts) poly.add(PointF(bb.float, bb.float))
            if (score >= confThreshold) {
                // Name from the handshake labels; fall back to DC_CLASSES by id, else a synthetic name.
                val name = labels.getOrNull(cls) ?: DC_CLASSES.getOrNull(cls) ?: "class $cls"
                results.add(DetectionResult(name, score, poly, cls))
            }
        }
        return results
    }

    fun processFrame(imageProxy: ImageProxy, onResult: (List<DetectionResult>) -> Unit) {
        if (!isReady) { imageProxy.close(); return }
        try {
            val bmp = imageProxyToBitmap(imageProxy)
            val scaled = Bitmap.createScaledBitmap(bmp, inputSize, inputSize, true)
            val results = infer(scaled)
            imageProxy.close()
            onResult(results)
        } catch (e: Exception) {
            e.printStackTrace(); imageProxy.close(); onResult(emptyList())
        }
    }

    /** Run one bitmap through the NPU; returns (detections, round-trip latency ms). */
    fun inferTimed(bitmap: Bitmap): Pair<List<DetectionResult>, Long> {
        if (!isReady) return Pair(emptyList(), 0L)
        val scaled = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val t0 = System.nanoTime()
        val res = infer(scaled)
        return Pair(res, (System.nanoTime() - t0) / 1_000_000)
    }

    /** Debug: run inference on a bundled asset image through the same NPU path. */
    fun processTestAsset(context: Context, assetName: String, onResult: (List<DetectionResult>) -> Unit) {
        if (!isReady) { onResult(emptyList()); return }
        Thread {
            try {
                val bmp = context.assets.open(assetName).use { BitmapFactory.decodeStream(it) }
                val scaled = Bitmap.createScaledBitmap(bmp, inputSize, inputSize, true)
                onResult(infer(scaled))
            } catch (e: Exception) {
                e.printStackTrace(); onResult(emptyList())
            }
        }.start()
    }

    /**
     * Convert an RGBA_8888 ImageProxy (ImageAnalysis output format) to an upright Bitmap.
     * Handles the row-stride padding and rotates by the sensor orientation so the model
     * sees the same upright RGB the training/test images use. (The previous YUV_420_888
     * path ignored rowStride/pixelStride and rotation -> garbled, sideways frames.)
     */
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val plane = imageProxy.planes[0]
        val buffer = plane.buffer.apply { rewind() }
        val pixelStride = plane.pixelStride                 // 4 for RGBA_8888
        val rowStride = plane.rowStride                     // may exceed width*4 (padding)
        val rowPadding = rowStride - pixelStride * imageProxy.width
        val padded = Bitmap.createBitmap(
            imageProxy.width + rowPadding / pixelStride,
            imageProxy.height,
            Bitmap.Config.ARGB_8888
        )
        padded.copyPixelsFromBuffer(buffer)
        val cropped = if (rowPadding == 0) padded
            else Bitmap.createBitmap(padded, 0, 0, imageProxy.width, imageProxy.height)
        val rot = imageProxy.imageInfo.rotationDegrees
        if (rot == 0) return cropped
        val m = android.graphics.Matrix().apply { postRotate(rot.toFloat()) }
        return Bitmap.createBitmap(cropped, 0, 0, cropped.width, cropped.height, m, true)
    }

    fun shutdown() { synchronized(ioLock) { closeSocketLocked() }; isReady = false }
}

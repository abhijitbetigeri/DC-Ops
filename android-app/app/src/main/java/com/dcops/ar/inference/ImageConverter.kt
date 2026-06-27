package com.dcops.ar.inference

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

/**
 * Converts a CameraX [ImageProxy] (YUV_420_888) into the tensor format a YOLO
 * `.pte` expects: NCHW float32, [inputSize × inputSize], RGB, normalized 0..1.
 *
 * This is real preprocessing the [ExecuTorchModelManager] will call directly.
 * It is intentionally self-contained (no native deps) so it can be unit-tested
 * and validated before the model lands.
 */
object ImageConverter {

    /** Decode a YUV_420_888 [ImageProxy] into an ARGB [Bitmap]. */
    fun toBitmap(image: ImageProxy): Bitmap {
        val nv21 = yuv420ToNv21(image)
        val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
        val bytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    /**
     * Resize [bitmap] to [size]×[size] and flatten to an NCHW float array
     * (channel-major: all R, then all G, then all B), normalized to 0..1.
     */
    fun toNchwFloat(bitmap: Bitmap, size: Int = 640): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, size, size, true)
        val pixels = IntArray(size * size)
        resized.getPixels(pixels, 0, size, 0, 0, size, size)

        val out = FloatArray(3 * size * size)
        val plane = size * size
        for (i in pixels.indices) {
            val p = pixels[i]
            out[i] = ((p shr 16) and 0xFF) / 255f             // R
            out[plane + i] = ((p shr 8) and 0xFF) / 255f       // G
            out[2 * plane + i] = (p and 0xFF) / 255f           // B
        }
        if (resized != bitmap) resized.recycle()
        return out
    }

    /** Pack the three YUV planes into a single NV21 byte array. */
    private fun yuv420ToNv21(image: ImageProxy): ByteArray {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val nv21 = ByteArray(ySize + ySize / 2)

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        // --- Y plane (copy row by row to respect rowStride) ---
        val yBuffer = yPlane.buffer
        val yRowStride = yPlane.rowStride
        var pos = 0
        if (yRowStride == width) {
            yBuffer.get(nv21, 0, ySize)
            pos = ySize
        } else {
            val row = ByteArray(yRowStride)
            for (r in 0 until height) {
                yBuffer.position(r * yRowStride)
                yBuffer.get(row, 0, yRowStride)
                System.arraycopy(row, 0, nv21, pos, width)
                pos += width
            }
        }

        // --- interleave V and U as VU (NV21) ---
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val chromaHeight = height / 2
        val chromaWidth = width / 2
        for (r in 0 until chromaHeight) {
            for (c in 0 until chromaWidth) {
                val uIndex = r * uRowStride + c * uPixelStride
                val vIndex = r * vPlane.rowStride + c * vPlane.pixelStride
                nv21[pos++] = vBuffer.get(vIndex)
                nv21[pos++] = uBuffer.get(uIndex)
            }
        }
        return nv21
    }
}

package com.dcops.ar.camera

import android.annotation.SuppressLint
import android.content.Context
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Manages CameraX setup: preview + frame analysis.
 *
 * Calls [onFrameAvailable] for every frame captured at [TARGET_RESOLUTION].
 * The callback is invoked on a background executor — the callee must call
 * [ImageProxy.close] when done to keep the camera pipeline flowing.
 */
class CameraManager(
    private val context: Context,
    private val previewView: PreviewView,
    private val onFrameAvailable: (ImageProxy) -> Unit
) {
    companion object {
        private val TARGET_RESOLUTION = Size(1280, 720)
    }

    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null

    @SuppressLint("RestrictedApi")
    fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()

            // Preview use-case
            val preview = Preview.Builder()
                .setTargetResolution(TARGET_RESOLUTION)
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            // Image analysis use-case — delivers frames to our callback
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(TARGET_RESOLUTION)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        onFrameAvailable(imageProxy)
                    }
                }

            // Select back camera
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind any previous use-cases and rebind
                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(
                    context as androidx.lifecycle.LifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
            } catch (e: Exception) {
                e.printStackTrace()
            }

        }, ContextCompat.getMainExecutor(context))
    }

    fun shutdown() {
        cameraProvider?.unbindAll()
        cameraExecutor.shutdown()
    }
}
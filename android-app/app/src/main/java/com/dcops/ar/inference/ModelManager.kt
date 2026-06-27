package com.dcops.ar.inference

import androidx.camera.core.ImageProxy

/**
 * The inference seam.
 *
 * The rest of the app talks ONLY to this interface, so the fake
 * [StubModelManager] and the real [ExecuTorchModelManager] are drop-in
 * interchangeable. Swapping from stub to the real `.pte` model is a one-line
 * change in [com.dcops.ar.MainActivity] — nothing else in the app moves.
 *
 * See ARCHITECTURE.md §5.3 (DetectionResult) and §5.4 (model artifact).
 */
interface ModelManager {

    /**
     * Load the model / warm up the backend.
     * @param onReady invoked (on a background thread) with `true` when ready.
     */
    fun init(onReady: (Boolean) -> Unit)

    /**
     * Process a single camera frame.
     *
     * The implementation MUST call [ImageProxy.close] when finished so CameraX
     * can deliver the next frame. Results use normalized image coordinates.
     */
    fun processFrame(imageProxy: ImageProxy, onResult: (List<DetectionResult>) -> Unit)

    /** Release any native resources (ExecuTorch module / QNN context). */
    fun shutdown()
}

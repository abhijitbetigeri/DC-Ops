package com.dcops.ar.inference

import android.content.Context
import androidx.camera.core.ImageProxy

/**
 * Real [ModelManager] backed by the ExecuTorch runtime on the Snapdragon NPU
 * (QNN backend). This is a **skeleton** — the wiring is laid out and documented
 * so swapping the stub for the real model is a focused, mechanical change once
 * the ExecuTorch AAR and `yolov8n_dcops_int8.pte` are available.
 *
 * To activate (Stream A, milestone M2):
 *   1. Add the ExecuTorch AAR to app/build.gradle.kts (see ARCHITECTURE.md §5.4).
 *   2. Place `yolov8n_dcops_int8.pte` in app/src/main/assets/models/.
 *   3. Implement the three TODOs below.
 *   4. In MainActivity, swap `StubModelManager()` → `ExecuTorchModelManager(this)`.
 */
class ExecuTorchModelManager(
    @Suppress("unused") private val context: Context,
    private val assetModelPath: String = "models/yolov8n_dcops_int8.pte",
    private val inputSize: Int = 640,
    private val scoreThreshold: Float = 0.4f
) : ModelManager {

    // private var module: org.pytorch.executorch.Module? = null

    override fun init(onReady: (Boolean) -> Unit) {
        Thread {
            // TODO(M2): copy the .pte out of assets and load it via the ExecuTorch AAR:
            //   val path = AssetCopier.copy(context, assetModelPath)
            //   module = Module.load(path)            // initializes the QNN backend
            onReady(false) // not implemented yet — keep using StubModelManager
        }.start()
    }

    override fun processFrame(imageProxy: ImageProxy, onResult: (List<DetectionResult>) -> Unit) {
        try {
            // val input = ImageConverter.toNchwFloat(ImageConverter.toBitmap(imageProxy), inputSize)
            // TODO(M2): val output = module!!.forward(EValue.from(Tensor.fromBlob(input, longArrayOf(1,3,inputSize,inputSize))))
            // TODO(M2): val detections = YoloDecoder.decode(output, scoreThreshold)  // → List<DetectionResult>
            onResult(emptyList())
        } finally {
            imageProxy.close()
        }
    }

    override fun shutdown() {
        // module?.destroy(); module = null
    }
}

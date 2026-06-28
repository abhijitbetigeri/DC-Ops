package com.dcops.ar

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.dcops.ar.databinding.ActivityMainBinding
import com.dcops.ar.inference.ModelManager
import com.dcops.ar.camera.CameraManager

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraManager: CameraManager
    private lateinit var modelManager: ModelManager
    private var useQnn = false
    private var frameCount = 0L
    private var lastFpsTime = System.nanoTime()

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startCamera()
        } else {
            Toast.makeText(this, R.string.camera_permission_denied, Toast.LENGTH_LONG).show()
            binding.statusText.text = getString(R.string.camera_permission_denied)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        modelManager = ModelManager()
        modelManager.init(this) { ready ->
            runOnUiThread {
                binding.statusText.text = if (ready) getString(R.string.status_ready) else "Model load failed"
                binding.backendLabel.text = "XNNPACK (CPU)"
            }
        }

        // Backend toggle: CPU <-> NPU
        binding.backendSwitch.setOnCheckedChangeListener { _, isChecked ->
            useQnn = isChecked
            val modelFile = if (useQnn) ModelManager.QNN_MODEL_FILENAME else ModelManager.MODEL_FILENAME
            binding.backendLabel.text = if (useQnn) "QNN HTP (NPU)" else "XNNPACK (CPU)"
            binding.backendLabel.setTextColor(
                if (useQnn) 0xFF00E676.toInt() else 0xFFFFEB3B.toInt()
            )
            binding.statusText.text = "Switching model..."

            modelManager.switchModel(this, modelFile) { ready ->
                runOnUiThread {
                    binding.statusText.text = if (ready) "Model loaded" else "Model load failed"
                }
            }
        }

        cameraManager = CameraManager(
            context = this,
            previewView = binding.previewView,
            onFrameAvailable = { imageProxy ->
                val startTime = System.nanoTime()
                modelManager.processFrame(imageProxy) { results ->
                    val inferenceMs = (System.nanoTime() - startTime) / 1_000_000.0
                    frameCount++

                    runOnUiThread {
                        binding.overlayView.updateDetections(results)
                        binding.latencyText.text = "${inferenceMs.toInt()}ms"

                        // Calculate FPS every 10 frames
                        if (frameCount % 10 == 0L) {
                            val now = System.nanoTime()
                            val elapsed = (now - lastFpsTime) / 1_000_000_000.0
                            val fps = 10.0 / elapsed
                            binding.fpsText.text = "${fps.toInt()} FPS"
                            lastFpsTime = now
                        }

                        binding.statusText.text = "${results.size} detections"
                    }
                }
            }
        )

        if (hasCameraPermission()) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        cameraManager.startCamera()
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraManager.shutdown()
        modelManager.shutdown()
    }
}

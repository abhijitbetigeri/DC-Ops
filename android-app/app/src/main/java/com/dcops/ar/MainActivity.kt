package com.dcops.ar

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.dcops.ar.databinding.ActivityMainBinding
import com.dcops.ar.inference.DetectionResult
import com.dcops.ar.inference.ModelManager
import com.dcops.ar.camera.CameraManager

/**
 * Main activity for the DC-Ops AR app.
 *
 * - Requests camera permission
 * - Starts CameraX preview via [CameraManager]
 * - Feeds frames to [ModelManager] (stub for now)
 * - Receives polygon detection results and forwards them to the overlay
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraManager: CameraManager
    private lateinit var modelManager: ModelManager

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
                binding.statusText.text = if (ready) {
                    getString(R.string.status_ready)
                } else {
                    "Model load failed"
                }
            }
        }

        cameraManager = CameraManager(
            context = this,
            previewView = binding.previewView,
            onFrameAvailable = { imageProxy ->
                // Pass the frame to the model manager (stub for now)
                modelManager.processFrame(imageProxy) { results ->
                    runOnUiThread {
                        binding.overlayView.updateDetections(results)
                        binding.statusText.text = getString(R.string.status_processing)
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
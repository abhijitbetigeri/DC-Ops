package com.dcops.ar

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.dcops.ar.camera.CameraManager
import com.dcops.ar.data.Finding
import com.dcops.ar.databinding.ActivityMainBinding
import com.dcops.ar.inference.DetectionResult
import com.dcops.ar.inference.ModelManager
import com.dcops.ar.inference.StubModelManager
import com.dcops.ar.ui.AuditLogActivity
import com.dcops.ar.ui.SettingsManager
import kotlinx.coroutines.launch

/**
 * Main AR screen.
 *
 * - Requests camera permission and starts the CameraX preview ([CameraManager]).
 * - Feeds frames to the [ModelManager] seam ([StubModelManager] today; swap to
 *   ExecuTorchModelManager at M2 — one line).
 * - Filters detections by the confidence threshold, draws them on the overlay.
 * - Capture button logs the current detections to the on-device audit log.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraManager: CameraManager
    private lateinit var modelManager: ModelManager
    private lateinit var settings: SettingsManager

    @Volatile
    private var latestDetections: List<DetectionResult> = emptyList()

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

        settings = SettingsManager(this)

        modelManager = StubModelManager()
        modelManager.init { ready ->
            runOnUiThread {
                binding.statusText.text =
                    getString(if (ready) R.string.status_ready else R.string.status_waiting)
            }
        }

        cameraManager = CameraManager(
            context = this,
            previewView = binding.previewView,
            onFrameAvailable = { imageProxy ->
                modelManager.processFrame(imageProxy) { results ->
                    val filtered = results.filter { it.score >= settings.confidenceThreshold }
                    latestDetections = filtered
                    runOnUiThread { onDetections(filtered) }
                }
            }
        )

        binding.captureButton.setOnClickListener { capture() }
        binding.logButton.setOnClickListener {
            startActivity(Intent(this, AuditLogActivity::class.java))
        }
        binding.settingsButton.setOnClickListener { showSettings() }

        if (hasCameraPermission()) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun onDetections(detections: List<DetectionResult>) {
        binding.overlayView.updateDetections(detections)
        binding.detectionChip.text =
            resources.getQuantityString(R.plurals.detections_count, detections.size, detections.size)
        binding.statusText.text = getString(R.string.status_processing)
    }

    /** Log the current on-screen detections to the audit log. */
    private fun capture() {
        val snapshot = latestDetections
        if (snapshot.isEmpty()) {
            Toast.makeText(this, R.string.capture_none, Toast.LENGTH_SHORT).show()
            return
        }
        val now = System.currentTimeMillis()
        val repo = (application as DcOpsApp).findingsRepository
        lifecycleScope.launch {
            snapshot.forEach { repo.add(Finding.from(it, now)) }
            Toast.makeText(
                this@MainActivity,
                resources.getQuantityString(R.plurals.capture_saved, snapshot.size, snapshot.size),
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    private fun showSettings() {
        val view = layoutInflater.inflate(R.layout.dialog_settings, null)
        val seek = view.findViewById<SeekBar>(R.id.confidenceSeek)
        val value = view.findViewById<TextView>(R.id.confidenceValue)
        val initial = (settings.confidenceThreshold * 100).toInt()
        seek.progress = initial
        value.text = getString(R.string.percent_fmt, initial)
        seek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar, progress: Int, fromUser: Boolean) {
                value.text = getString(R.string.percent_fmt, progress)
            }

            override fun onStartTrackingTouch(s: SeekBar) {}
            override fun onStopTrackingTouch(s: SeekBar) {}
        })
        AlertDialog.Builder(this)
            .setTitle(R.string.settings_title)
            .setView(view)
            .setPositiveButton(R.string.save) { _, _ ->
                settings.confidenceThreshold = seek.progress / 100f
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun startCamera() = cameraManager.startCamera()

    private fun hasCameraPermission(): Boolean = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    override fun onDestroy() {
        super.onDestroy()
        cameraManager.shutdown()
        modelManager.shutdown()
    }
}

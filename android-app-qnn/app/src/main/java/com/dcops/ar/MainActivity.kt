package com.dcops.ar

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.SeekBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
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

    // Debug: tap the screen to cycle the bundled test images instead of the camera.
    private var testMode = false
    private var testThread: Thread? = null

    // Detection persistence: hold the last non-empty detections briefly so sporadic
    // live detections (especially when pointing at a screen) don't flicker on/off.
    private var heldResults: List<DetectionResult> = emptyList()
    private var heldAtMs = 0L
    private val holdMs = 500L
    private fun holdDetections(results: List<DetectionResult>): List<DetectionResult> {
        val now = android.os.SystemClock.uptimeMillis()
        if (results.isNotEmpty()) { heldResults = results; heldAtMs = now; return results }
        if (now - heldAtMs < holdMs) return heldResults
        heldResults = emptyList(); return emptyList()
    }

    private companion object {
        val TEST_ASSETS = listOf(
            "test_rack.jpg", "test_1.jpg", "test_2.jpg", "test_3.jpg",
            "test_4.jpg", "test_5.jpg", "test_6.jpg"
        )
    }

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

        enableImmersiveMode()
        // Feed the real status-bar height to the overlay so labels never hide under it.
        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { _, insets ->
            binding.overlayView.topInset =
                insets.getInsets(WindowInsetsCompat.Type.statusBars()).top.toFloat()
            insets
        }

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
                if (testMode) {
                    imageProxy.close()   // paused while showing the test image
                } else {
                    modelManager.processFrame(imageProxy) { results ->
                        runOnUiThread {
                            binding.overlayView.updateDetections(holdDetections(results))
                            binding.statusText.text = getString(R.string.status_processing)
                        }
                    }
                }
            }
        )

        // Tap the screen OR press Volume Down to toggle the test-image diagnostic.
        binding.overlayView.isClickable = true
        binding.overlayView.setOnClickListener { toggleTestMode() }
        binding.statusText.setOnClickListener { toggleTestMode() }

        // Confidence-threshold slider: only show detections at/above this confidence.
        binding.thresholdSeek.progress = (ModelManager.DEFAULT_CONF * 100).toInt()
        binding.thresholdLabel.text = "Min confidence: ${(ModelManager.DEFAULT_CONF * 100).toInt()}%"
        binding.thresholdSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                modelManager.confThreshold = progress / 100f
                binding.thresholdLabel.text = "Min confidence: $progress%"
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        // Server-floor slider (always visible): tunes the NPU-side conf floor live.
        // Drag down to surface more (weaker) detections, up to be stricter.
        binding.floorSeek.progress = (ModelManager.DEFAULT_SERVER_FLOOR * 100).toInt()
        binding.floorLabel.text = "Server floor: ${(ModelManager.DEFAULT_SERVER_FLOOR * 100).toInt()}%"
        binding.floorSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                modelManager.serverFloor = progress / 100f
                binding.floorLabel.text = "Server floor: $progress%"
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        if (hasCameraPermission()) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onKeyDown(keyCode: Int, event: android.view.KeyEvent?): Boolean {
        if (keyCode == android.view.KeyEvent.KEYCODE_VOLUME_DOWN ||
            keyCode == android.view.KeyEvent.KEYCODE_VOLUME_UP) {
            toggleTestMode()
            return true
        }
        return super.onKeyDown(keyCode, event)
    }

    private fun toggleTestMode() {
        testMode = !testMode
        if (testMode) {
            binding.testImageView.visibility = android.view.View.VISIBLE
            startTestCycle()
        } else {
            testThread?.interrupt(); testThread = null
            binding.testImageView.visibility = android.view.View.GONE
            binding.overlayView.updateDetections(emptyList())
            binding.statusText.text = getString(R.string.status_ready)
        }
    }

    /** Cycle through the bundled test images one-by-one, showing per-image NPU latency + fps. */
    private fun startTestCycle() {
        testThread?.interrupt()
        val t = Thread {
            var i = 0
            while (testMode && !Thread.currentThread().isInterrupted) {
                val asset = TEST_ASSETS[i % TEST_ASSETS.size]
                val idx = i % TEST_ASSETS.size + 1
                try {
                    val bmp = assets.open(asset).use { android.graphics.BitmapFactory.decodeStream(it) }
                    runOnUiThread { binding.testImageView.setImageBitmap(bmp) }
                    val (results, ms) = modelManager.inferTimed(bmp)
                    if (!testMode) break
                    val fps = if (ms > 0) (1000.0 / ms) else 0.0
                    runOnUiThread {
                        binding.overlayView.updateDetections(results)
                        binding.statusText.text =
                            "img $idx/${TEST_ASSETS.size}  •  ${ms} ms  •  %.1f fps  •  ${results.size} det".format(fps)
                    }
                    Thread.sleep(700)   // hold each image briefly so it's watchable
                } catch (e: InterruptedException) {
                    break
                } catch (e: Exception) {
                    e.printStackTrace()
                }
                i++
            }
        }
        testThread = t
        t.start()
    }

    /** Hide the navigation bar (back / home / recents) for a clean full-screen AR view. */
    private fun enableImmersiveMode() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val controller = WindowInsetsControllerCompat(window, binding.root)
        controller.hide(WindowInsetsCompat.Type.navigationBars())
        controller.systemBarsBehavior =
            WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) enableImmersiveMode()   // re-hide after dialogs / focus returns
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
        testMode = false
        testThread?.interrupt(); testThread = null
        cameraManager.shutdown()
        modelManager.shutdown()
    }
}
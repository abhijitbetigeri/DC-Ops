package com.dcops.ar

import android.Manifest
import android.content.res.ColorStateList
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
import com.dcops.ar.ui.DetectionMissionStyle
import com.google.android.material.card.MaterialCardView

/**
 * Main activity for the DC-Ops AR app.
 *
 * - Requests camera permission
 * - Starts CameraX preview via [CameraManager]
 * - Feeds frames to [ModelManager] (stub for now)
 * - Receives polygon detection results and forwards them to the overlay
 */
class MainActivity : AppCompatActivity() {

    private enum class OperatorAction(
        val stageResId: Int,
        val statusResId: Int
    ) {
        CONFIRM_LED(R.string.focus_stage_confirmed, R.string.status_confirmed_pattern),
        CAPTURE_OCR(R.string.focus_stage_confirmed, R.string.status_ocr_pattern),
        FLAG_CABLE(R.string.focus_stage_confirmed, R.string.status_flagged_pattern),
        SAVE_LOG(R.string.focus_stage_logged, R.string.status_logged_pattern)
    }

    private data class OperatorCommit(
        val detectionLabel: String,
        val action: OperatorAction
    )

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraManager: CameraManager
    private lateinit var modelManager: ModelManager
    private var focusDetection: DetectionResult? = null
    private var operatorCommit: OperatorCommit? = null
    private var modelReady = false

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

        binding.liveChip.backgroundTintList = ColorStateList.valueOf(
            ContextCompat.getColor(this, R.color.mission_green)
        )
        binding.computeChip.backgroundTintList = ColorStateList.valueOf(
            ContextCompat.getColor(this, R.color.mission_chip)
        )
        bindActionRail()
        renderTelemetry(emptyList(), null)

        modelManager = ModelManager()
        modelManager.init(this) { ready ->
            runOnUiThread {
                modelReady = ready
                binding.statusText.text = if (ready) {
                    getString(R.string.status_ready)
                } else {
                    getString(R.string.status_error_model)
                }
                renderTelemetry(emptyList(), focusDetection)
            }
        }

        cameraManager = CameraManager(
            context = this,
            previewView = binding.previewView,
            onFrameAvailable = { imageProxy ->
                modelManager.processFrame(imageProxy) { results ->
                    runOnUiThread {
                        renderTelemetry(results, DetectionMissionStyle.pickFocus(results))
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

    private fun bindActionRail() {
        binding.actionConfirmLed.setOnClickListener { commitFocus(OperatorAction.CONFIRM_LED) }
        binding.actionCaptureOcr.setOnClickListener { commitFocus(OperatorAction.CAPTURE_OCR) }
        binding.actionFlagCable.setOnClickListener { commitFocus(OperatorAction.FLAG_CABLE) }
        binding.actionSaveLog.setOnClickListener { commitFocus(OperatorAction.SAVE_LOG) }
    }

    private fun commitFocus(action: OperatorAction) {
        val currentFocus = focusDetection
        if (currentFocus == null) {
            Toast.makeText(this, R.string.status_waiting_for_target, Toast.LENGTH_SHORT).show()
            binding.statusText.text = getString(R.string.status_waiting_for_target)
            return
        }

        operatorCommit = OperatorCommit(currentFocus.label, action)
        binding.statusText.text = getString(
            action.statusResId,
            DetectionMissionStyle.displayLabel(this, currentFocus)
        )
        renderFocusCard(currentFocus)
    }

    private fun renderTelemetry(results: List<DetectionResult>, newFocus: DetectionResult?) {
        if (focusDetection?.label != newFocus?.label) {
            operatorCommit = null
        }
        focusDetection = newFocus

        binding.overlayView.updateDetections(results, newFocus)
        binding.targetsMetricValue.text = if (results.isEmpty()) {
            getString(R.string.metric_targets_zero)
        } else {
            getString(R.string.metric_targets_pattern, results.size)
        }
        binding.confidenceMetricValue.text = newFocus?.let {
            getString(R.string.metric_confidence_pattern, (it.score * 100).toInt())
        } ?: getString(R.string.metric_confidence_idle)
        binding.nextStepMetricValue.text = DetectionMissionStyle.nextStep(this, newFocus)
        binding.missionSubtitle.text = missionSubtitleFor(newFocus)

        if (operatorCommit == null) {
            binding.statusText.text = statusLineFor(results, newFocus)
        }

        renderFocusCard(newFocus)
    }

    private fun renderFocusCard(detection: DetectionResult?) {
        val accentColor = DetectionMissionStyle.accentColor(this, detection)
        val commit = operatorCommit?.takeIf { it.detectionLabel == detection?.label }

        binding.focusCard.strokeColor = accentColor
        binding.focusStateChip.text = DetectionMissionStyle.stateLabel(this, detection)
        binding.focusStateChip.backgroundTintList = ColorStateList.valueOf(accentColor)
        binding.focusStateChip.setTextColor(ContextCompat.getColor(this, R.color.mission_canvas))
        binding.focusStage.text = commit?.let { getString(it.action.stageResId) }
            ?: DetectionMissionStyle.stageLabel(this, detection)
        binding.focusTitle.text = DetectionMissionStyle.displayLabel(this, detection)
        binding.focusSummary.text = DetectionMissionStyle.summary(this, detection)
        binding.focusConfidence.text = detection?.let {
            getString(R.string.metric_confidence_pattern, (it.score * 100).toInt())
        } ?: getString(R.string.metric_confidence_idle)
        binding.focusConfidence.setTextColor(accentColor)
        binding.focusDetail.text = if (commit == null) {
            DetectionMissionStyle.detail(this, detection)
        } else {
            getString(R.string.focus_detail_committed_pattern, getString(commit.action.stageResId))
        }

        styleMetricCard(binding.targetsMetricCard, ContextCompat.getColor(this, R.color.mission_cyan))
        styleMetricCard(binding.confidenceMetricCard, accentColor)
        styleMetricCard(binding.nextStepMetricCard, accentColor)
    }

    private fun missionSubtitleFor(detection: DetectionResult?): String {
        if (!modelReady) return getString(R.string.mission_subtitle_loading)

        return getString(
            when (DetectionMissionStyle.urgencyFor(detection)) {
                com.dcops.ar.ui.DetectionUrgency.HEALTHY -> R.string.mission_subtitle_ready
                com.dcops.ar.ui.DetectionUrgency.REVIEW -> R.string.mission_subtitle_review
                com.dcops.ar.ui.DetectionUrgency.CRITICAL -> R.string.mission_subtitle_action
            }
        )
    }

    private fun statusLineFor(results: List<DetectionResult>, detection: DetectionResult?): String {
        if (!modelReady) return getString(R.string.status_booting)
        if (detection == null) return getString(R.string.status_processing)
        return getString(
            R.string.status_focus_pattern,
            DetectionMissionStyle.displayLabel(this, detection),
            results.size
        )
    }

    private fun styleMetricCard(cardView: MaterialCardView, accentColor: Int) {
        cardView.strokeColor = accentColor
    }
}
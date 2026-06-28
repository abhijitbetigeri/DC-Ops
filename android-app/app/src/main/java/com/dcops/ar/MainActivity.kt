package com.dcops.ar

import android.Manifest
import android.content.res.ColorStateList
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.dcops.ar.camera.CameraManager
import com.dcops.ar.databinding.ActivityMainBinding
import com.dcops.ar.inference.DetectionResult
import com.dcops.ar.inference.ModelManager
import com.dcops.ar.overlay.CableMatchVisualization
import com.dcops.ar.ui.DetectionMissionStyle
import com.google.android.material.card.MaterialCardView

/**
 * Main activity for the DC-Ops AR app.
 *
 * Camera-backed mission-control screen for the DC-Ops AR app.
 */
class MainActivity : AppCompatActivity() {

    private enum class InspectionMode {
        SERVER_SCAN,
        CABLE_MATCH
    }

    private enum class CableStep(
        val cableResId: Int,
        val portResId: Int,
        val accentColorResId: Int
    ) {
        POWER(R.string.cable_power, R.string.port_power, R.color.mission_red),
        ETHERNET(R.string.cable_ethernet, R.string.port_ethernet, R.color.mission_cyan),
        USB_C(R.string.cable_usb_c, R.string.port_usb_c, R.color.mission_green)
    }

    private data class TelemetryFrame(
        val detections: List<DetectionResult>,
        val focus: DetectionResult?,
        val matchVisualization: CableMatchVisualization?,
        val statusText: String,
        val nextStepText: String,
        val subtitleText: String
    )

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraManager: CameraManager
    private lateinit var modelManager: ModelManager
    private var focusDetection: DetectionResult? = null
    private var modelReady = false
    private var inspectionMode = InspectionMode.SERVER_SCAN
    private var cableStepIndex = 0
    private var cableSeenForStep = false
    private var lastRawResults: List<DetectionResult> = emptyList()

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
        updateModeButton()
        renderTelemetry(emptyList())

        modelManager = ModelManager()
        modelManager.init(this) { ready ->
            runOnUiThread {
                modelReady = ready
                binding.statusText.text = if (ready) {
                    getString(R.string.status_ready)
                } else {
                    getString(R.string.status_error_model)
                }
                renderTelemetry(lastRawResults)
            }
        }

        cameraManager = CameraManager(
            context = this,
            previewView = binding.previewView,
            onFrameAvailable = { imageProxy ->
                modelManager.processFrame(imageProxy) { results ->
                    runOnUiThread {
                        renderTelemetry(results)
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

    override fun onDestroy() {
        super.onDestroy()
        if (::cameraManager.isInitialized) cameraManager.shutdown()
        if (::modelManager.isInitialized) modelManager.shutdown()
    }

    private fun startCamera() {
        cameraManager.startCamera()
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun bindActionRail() {
        binding.modeToggleButton.setOnClickListener { toggleInspectionMode() }
        binding.nextCableButton.setOnClickListener { advanceToNextCable() }
    }

    private fun toggleInspectionMode() {
        inspectionMode = when (inspectionMode) {
            InspectionMode.SERVER_SCAN -> InspectionMode.CABLE_MATCH
            InspectionMode.CABLE_MATCH -> InspectionMode.SERVER_SCAN
        }
        if (inspectionMode == InspectionMode.CABLE_MATCH) {
            resetCableSequence()
        }
        updateModeButton()
        renderTelemetry(lastRawResults)
    }

    private fun updateModeButton() {
        val modeLabel = getString(
            when (inspectionMode) {
                InspectionMode.SERVER_SCAN -> R.string.mode_server_scan
                InspectionMode.CABLE_MATCH -> R.string.mode_cable_match
            }
        )
        val tintColor = ContextCompat.getColor(
            this,
            when (inspectionMode) {
                InspectionMode.SERVER_SCAN -> R.color.mission_cyan
                InspectionMode.CABLE_MATCH -> R.color.mission_green
            }
        )
        binding.modeToggleButton.text = getString(R.string.action_toggle_mode_pattern, modeLabel)
        binding.modeToggleButton.backgroundTintList = ColorStateList.valueOf(tintColor)
        updateNextCableButton()
    }

    private fun resetCableSequence() {
        cableStepIndex = 0
        cableSeenForStep = false
    }

    private fun advanceToNextCable() {
        if (inspectionMode != InspectionMode.CABLE_MATCH) return
        cableStepIndex = (cableStepIndex + 1) % CableStep.entries.size
        cableSeenForStep = false
        updateNextCableButton()
        renderTelemetry(lastRawResults)
    }

    private fun updateNextCableButton() {
        val nextStep = CableStep.entries[(cableStepIndex + 1) % CableStep.entries.size]
        binding.nextCableButton.text = getString(
            R.string.action_next_cable_pattern,
            getString(nextStep.cableResId)
        )
        val enabled = inspectionMode == InspectionMode.CABLE_MATCH
        binding.nextCableButton.isEnabled = enabled
        binding.nextCableButton.visibility = if (enabled) View.VISIBLE else View.GONE
        binding.nextCableButton.alpha = if (enabled) 1f else 0.45f
        binding.nextCableButton.backgroundTintList = ColorStateList.valueOf(
            ContextCompat.getColor(
                this,
                if (enabled) R.color.mission_green else R.color.mission_chip
            )
        )
    }

    private fun updateCableSeenState(cable: DetectionResult?) {
        if (cable != null) {
            cableSeenForStep = true
        }
    }

    private fun renderTelemetry(results: List<DetectionResult>) {
        lastRawResults = results
        val frame = buildTelemetryFrame(results)

        focusDetection = frame.focus

        binding.overlayView.updateDetections(
            frame.detections,
            frame.focus,
            frame.matchVisualization
        )
        binding.targetsMetricValue.text = if (frame.detections.isEmpty()) {
            getString(R.string.metric_targets_zero)
        } else {
            getString(R.string.metric_targets_pattern, frame.detections.size)
        }
        binding.confidenceMetricValue.text = frame.focus?.let {
            getString(R.string.metric_confidence_pattern, (it.score * 100).toInt())
        } ?: getString(R.string.metric_confidence_idle)
        binding.nextStepMetricValue.text = frame.nextStepText
        binding.missionSubtitle.text = frame.subtitleText

        binding.statusText.text = frame.statusText

        if (inspectionMode == InspectionMode.CABLE_MATCH) {
            renderCableFocusCard(frame)
        } else {
            renderFocusCard(frame.focus)
        }
    }

    private fun buildTelemetryFrame(rawResults: List<DetectionResult>): TelemetryFrame {
        return when (inspectionMode) {
            InspectionMode.SERVER_SCAN -> buildServerScanFrame(rawResults)
            InspectionMode.CABLE_MATCH -> buildCableMatchFrame(rawResults)
        }
    }

    private fun buildServerScanFrame(rawResults: List<DetectionResult>): TelemetryFrame {
        val focus = DetectionMissionStyle.pickFocus(rawResults)
        return TelemetryFrame(
            detections = rawResults,
            focus = focus,
            matchVisualization = null,
            statusText = getString(R.string.status_scan_components_pattern, rawResults.size),
            nextStepText = DetectionMissionStyle.nextStep(this, focus),
            subtitleText = missionSubtitleFor(focus)
        )
    }

    private fun buildCableMatchFrame(rawResults: List<DetectionResult>): TelemetryFrame {
        val step = currentCableStep()
        val port = rawResults.firstOrNull { matchesPort(it, step) }
        val cable = rawResults.firstOrNull { matchesCable(it, step) }
        val hasMatch = cable != null && port != null

        updateCableSeenState(cable)

        val cableLabel = getString(step.cableResId)
        val portLabel = getString(step.portResId)
        val accentColor = ContextCompat.getColor(this, step.accentColorResId)
        val matchVisualization = CableMatchVisualization(
            cableDetection = cable,
            portDetection = port,
            cableLabel = cableLabel,
            portLabel = portLabel,
            activeColor = accentColor
        )

        return TelemetryFrame(
            detections = rawResults,
            focus = cable ?: port,
            matchVisualization = matchVisualization,
            statusText = if (hasMatch) {
                getString(R.string.status_match_locked_pattern, cableLabel, portLabel)
            } else {
                getString(R.string.status_match_waiting_pattern, cableLabel)
            },
            nextStepText = if (hasMatch) {
                getString(R.string.action_next_cable)
            } else {
                getString(R.string.metric_next_step_show_cable_pattern, cableLabel)
            },
            subtitleText = getString(R.string.mission_subtitle_cable_match)
        )
    }

    private fun currentCableStep(): CableStep = CableStep.entries[cableStepIndex]

    private fun matchesCable(detection: DetectionResult, step: CableStep): Boolean {
        val label = detection.label.lowercase()
        val isGenericCable = detection.classId == 5 || label == "cable"
        return when (step) {
            CableStep.POWER -> "power cable" in label || isGenericCable
            CableStep.ETHERNET -> "ethernet" in label || "network cable" in label || isGenericCable
            CableStep.USB_C -> "usb" in label || "type-c" in label || isGenericCable
        }
    }

    private fun matchesPort(detection: DetectionResult, step: CableStep): Boolean {
        val label = detection.label.lowercase()
        return when (step) {
            CableStep.POWER -> "power connector" in label || "power shelf" in label
            CableStep.ETHERNET -> "network port" in label
            CableStep.USB_C -> "usb" in label || "management port" in label
        }
    }

    private fun renderCableFocusCard(frame: TelemetryFrame) {
        val match = frame.matchVisualization
        val accentColor = match?.activeColor ?: ContextCompat.getColor(this, R.color.mission_cyan)
        val hasMatch = match?.cableDetection != null && match.portDetection != null
        val cableLabel = match?.cableLabel ?: getString(currentCableStep().cableResId)
        val portLabel = match?.portLabel ?: getString(currentCableStep().portResId)

        binding.focusCard.strokeColor = accentColor
        binding.focusStateChip.text = getString(
            if (hasMatch) R.string.focus_state_match else R.string.focus_state_next_cable
        )
        binding.focusStateChip.backgroundTintList = ColorStateList.valueOf(accentColor)
        binding.focusStateChip.setTextColor(ContextCompat.getColor(this, R.color.mission_canvas))
        binding.focusStage.text = getString(R.string.focus_stage_cable_route)
        binding.focusTitle.text = if (hasMatch) {
            getString(R.string.focus_title_route_pattern, cableLabel, portLabel)
        } else {
            getString(R.string.focus_title_show_cable_pattern, cableLabel)
        }
        binding.focusSummary.text = if (hasMatch) {
            getString(R.string.focus_summary_match_locked_pattern, cableLabel, portLabel)
        } else {
            getString(R.string.focus_summary_match_waiting_pattern, cableLabel, portLabel)
        }
        binding.focusConfidence.text = frame.focus?.let {
            getString(R.string.metric_confidence_pattern, (it.score * 100).toInt())
        } ?: getString(R.string.metric_confidence_idle)
        binding.focusConfidence.setTextColor(accentColor)
        binding.focusDetail.text = getString(
            if (hasMatch) R.string.focus_detail_match_locked else R.string.focus_detail_match_waiting
        )

        styleMetricCard(binding.targetsMetricCard, ContextCompat.getColor(this, R.color.mission_cyan))
        styleMetricCard(binding.confidenceMetricCard, accentColor)
        styleMetricCard(binding.nextStepMetricCard, accentColor)
    }

    private fun renderFocusCard(detection: DetectionResult?) {
        val accentColor = DetectionMissionStyle.accentColor(this, detection)

        binding.focusCard.strokeColor = accentColor
        binding.focusStateChip.text = DetectionMissionStyle.stateLabel(this, detection)
        binding.focusStateChip.backgroundTintList = ColorStateList.valueOf(accentColor)
        binding.focusStateChip.setTextColor(ContextCompat.getColor(this, R.color.mission_canvas))
        binding.focusStage.text = DetectionMissionStyle.stageLabel(this, detection)
        binding.focusTitle.text = DetectionMissionStyle.displayLabel(this, detection)
        binding.focusSummary.text = DetectionMissionStyle.summary(this, detection)
        binding.focusConfidence.text = detection?.let {
            getString(R.string.metric_confidence_pattern, (it.score * 100).toInt())
        } ?: getString(R.string.metric_confidence_idle)
        binding.focusConfidence.setTextColor(accentColor)
        binding.focusDetail.text = DetectionMissionStyle.detail(this, detection)

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
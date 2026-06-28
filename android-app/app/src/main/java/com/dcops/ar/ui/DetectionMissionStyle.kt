package com.dcops.ar.ui

import android.content.Context
import androidx.annotation.ColorInt
import androidx.core.content.ContextCompat
import com.dcops.ar.R
import com.dcops.ar.inference.DetectionResult

enum class DetectionUrgency {
    HEALTHY,
    REVIEW,
    CRITICAL
}

object DetectionMissionStyle {

    fun pickFocus(results: List<DetectionResult>): DetectionResult? {
        return results.maxWithOrNull(
            compareBy<DetectionResult>({ urgencyWeight(urgencyFor(it)) }, { it.score })
        )
    }

    fun urgencyFor(detection: DetectionResult?): DetectionUrgency {
        if (detection == null) return DetectionUrgency.HEALTHY

        val label = detection.label.lowercase()
        return when {
            "cable" in label || "power" in label -> DetectionUrgency.CRITICAL
            "management port" in label || "network port" in label -> DetectionUrgency.REVIEW
            "label" in label -> DetectionUrgency.REVIEW
            "led" in label && detection.score < 0.65f -> DetectionUrgency.REVIEW
            detection.score < 0.45f -> DetectionUrgency.REVIEW
            else -> DetectionUrgency.HEALTHY
        }
    }

    fun displayLabel(context: Context, detection: DetectionResult?): String {
        return detection?.label
            ?.split(" ")
            ?.joinToString(" ") { token ->
                token.replaceFirstChar { ch ->
                    if (ch.isLowerCase()) ch.titlecase() else ch.toString()
                }
            }
            ?: context.getString(R.string.focus_title_idle)
    }

    fun stateLabel(context: Context, detection: DetectionResult?): String {
        if (detection == null) return context.getString(R.string.focus_state_standby)
        return context.getString(
            when (urgencyFor(detection)) {
                DetectionUrgency.HEALTHY -> R.string.focus_state_healthy
                DetectionUrgency.REVIEW -> R.string.focus_state_review
                DetectionUrgency.CRITICAL -> R.string.focus_state_critical
            }
        )
    }

    fun stageLabel(context: Context, detection: DetectionResult?): String {
        if (detection == null) return context.getString(R.string.focus_stage_ready)
        return context.getString(
            when (urgencyFor(detection)) {
                DetectionUrgency.HEALTHY -> R.string.focus_stage_confirmed
                DetectionUrgency.REVIEW -> R.string.focus_stage_detected
                DetectionUrgency.CRITICAL -> R.string.focus_stage_suggested
            }
        )
    }

    fun nextStep(context: Context, detection: DetectionResult?): String {
        if (detection == null) return context.getString(R.string.metric_next_step_idle)

        return context.getString(
            when (urgencyFor(detection)) {
                DetectionUrgency.HEALTHY -> R.string.metric_next_step_save_log
                DetectionUrgency.REVIEW -> R.string.metric_next_step_capture_ocr
                DetectionUrgency.CRITICAL -> {
                    if ("cable" in detection.label.lowercase()) {
                        R.string.metric_next_step_flag_cable
                    } else {
                        R.string.metric_next_step_confirm_led
                    }
                }
            }
        )
    }

    fun detail(context: Context, detection: DetectionResult?): String {
        if (detection == null) return context.getString(R.string.focus_detail_idle)

        return context.getString(
            when (urgencyFor(detection)) {
                DetectionUrgency.HEALTHY -> R.string.focus_detail_healthy
                DetectionUrgency.REVIEW -> R.string.focus_detail_review
                DetectionUrgency.CRITICAL -> R.string.focus_detail_critical
            },
            nextStep(context, detection)
        )
    }

    fun summary(context: Context, detection: DetectionResult?): String {
        if (detection == null) return context.getString(R.string.focus_summary_idle)
        return context.getString(
            R.string.focus_summary_pattern,
            displayLabel(context, detection),
            (detection.score * 100).toInt()
        )
    }

    @ColorInt
    fun accentColor(context: Context, detection: DetectionResult?): Int {
        if (detection == null) {
            return ContextCompat.getColor(context, R.color.mission_cyan)
        }

        return ContextCompat.getColor(
            context,
            when (urgencyFor(detection)) {
                DetectionUrgency.HEALTHY -> R.color.mission_green
                DetectionUrgency.REVIEW -> R.color.mission_amber
                DetectionUrgency.CRITICAL -> R.color.mission_red
            }
        )
    }

    private fun urgencyWeight(urgency: DetectionUrgency): Int {
        return when (urgency) {
            DetectionUrgency.HEALTHY -> 0
            DetectionUrgency.REVIEW -> 1
            DetectionUrgency.CRITICAL -> 2
        }
    }
}

package com.dcops.ar.ui

import android.content.Context

/**
 * Thin [android.content.SharedPreferences] wrapper for user-tunable settings.
 * Currently: the minimum confidence to surface/log a detection, and whether to
 * auto-log confident detections vs. tap-to-capture (ARCHITECTURE.md §9).
 */
class SettingsManager(context: Context) {

    private val prefs =
        context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    var confidenceThreshold: Float
        get() = prefs.getFloat(KEY_CONFIDENCE, DEFAULT_CONFIDENCE)
        set(value) = prefs.edit().putFloat(KEY_CONFIDENCE, value).apply()

    var autoCapture: Boolean
        get() = prefs.getBoolean(KEY_AUTO_CAPTURE, false)
        set(value) = prefs.edit().putBoolean(KEY_AUTO_CAPTURE, value).apply()

    companion object {
        private const val PREFS = "dcops_settings"
        private const val KEY_CONFIDENCE = "confidence_threshold"
        private const val KEY_AUTO_CAPTURE = "auto_capture"
        const val DEFAULT_CONFIDENCE = 0.5f
    }
}

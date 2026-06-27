package com.dcops.ar.inference

import android.graphics.Color

/**
 * The frozen DC-Ops class taxonomy — the single source of truth for
 * `classId ↔ label ↔ color`. Mirrors `models/classes.yaml` exactly.
 *
 * The YOLO model (Stream B) is trained to these exact integer IDs; the app
 * (Streams A/C) renders them. If you add or reorder a class, change BOTH this
 * enum and `models/classes.yaml` and tell the team — it is a contract.
 */
enum class DcClass(
    val id: Int,
    val key: String,
    val displayName: String,
    val colorHex: String
) {
    LED_GREEN(0, "led_green", "LED · Green", "#00E676"),
    LED_AMBER(1, "led_amber", "LED · Amber", "#FFC107"),
    LED_RED(2, "led_red", "LED · Red", "#FF1744"),
    LED_OFF(3, "led_off", "LED · Off", "#9E9E9E"),
    CABLE(4, "cable", "Cable", "#2979FF"),
    LABEL(5, "label", "Label / Serial", "#E040FB");

    val color: Int get() = Color.parseColor(colorHex)

    /** True for the LED status classes (used to drive the green✓/red✗ summary). */
    val isLed: Boolean get() = this == LED_GREEN || this == LED_AMBER || this == LED_RED || this == LED_OFF

    companion object {
        const val COUNT = 6
        private val byId = entries.associateBy { it.id }
        fun fromId(id: Int): DcClass? = byId[id]
    }
}

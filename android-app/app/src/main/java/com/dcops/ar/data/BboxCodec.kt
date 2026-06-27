package com.dcops.ar.data

import android.graphics.PointF

/**
 * Serializes a normalized polygon to/from a compact text form for SQLite and CSV.
 *
 * Format: `"x1,y1;x2,y2;..."` — deliberately dependency-free (no JSON lib) so it
 * is trivial to parse and unit-test.
 */
object BboxCodec {

    fun encode(points: List<PointF>): String =
        points.joinToString(";") { "${it.x},${it.y}" }

    fun decode(s: String): List<PointF> {
        if (s.isBlank()) return emptyList()
        return s.split(";").mapNotNull { pair ->
            val parts = pair.split(",")
            if (parts.size != 2) return@mapNotNull null
            val x = parts[0].toFloatOrNull() ?: return@mapNotNull null
            val y = parts[1].toFloatOrNull() ?: return@mapNotNull null
            PointF(x, y)
        }
    }
}

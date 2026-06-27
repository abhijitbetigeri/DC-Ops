package com.dcops.ar.data

/**
 * Serializes [Finding]s to CSV for the "export via secure channel" flow.
 * Pure/JVM-friendly (no Android deps) so it is unit-testable.
 */
object CsvExporter {

    private val HEADER = listOf(
        "id", "timestamp_ms", "class_id", "label", "score", "bbox", "serial", "image_ref"
    )

    fun toCsv(findings: List<Finding>): String = buildString {
        appendLine(HEADER.joinToString(","))
        for (f in findings) {
            appendLine(
                listOf(
                    f.id.toString(),
                    f.timestampMs.toString(),
                    f.classId.toString(),
                    escape(f.label),
                    f.score.toString(),
                    escape(BboxCodec.encode(f.bbox)),
                    escape(f.serial ?: ""),
                    escape(f.imageRef ?: "")
                ).joinToString(",")
            )
        }
    }

    /** Minimal RFC-4180 escaping: quote fields containing comma/quote/newline. */
    private fun escape(value: String): String =
        if (value.any { it == ',' || it == '"' || it == '\n' }) {
            "\"" + value.replace("\"", "\"\"") + "\""
        } else value
}

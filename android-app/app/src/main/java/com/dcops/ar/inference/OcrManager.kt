package com.dcops.ar.inference

import android.graphics.Bitmap

/**
 * OCR seam (ARCHITECTURE.md §5.7) — a STRETCH goal.
 *
 * Defined now so the detection → crop → OCR flow can be wired and the
 * [com.dcops.ar.data.Finding.serial] field populated, but only implemented if
 * the MVP is done. [StubOcrManager] keeps everything compiling and demo-able.
 */
interface OcrManager {
    /** Read alphanumeric text from a cropped label region. */
    suspend fun readText(crop: Bitmap): String
}

/** No-op OCR used until a real CRNN/PP-OCR `.pte` is wired in. */
class StubOcrManager : OcrManager {
    override suspend fun readText(crop: Bitmap): String = ""
}

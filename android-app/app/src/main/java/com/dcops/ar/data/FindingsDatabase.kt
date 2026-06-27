package com.dcops.ar.data

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

/**
 * Raw SQLite store for [Finding]s (ARCHITECTURE.md §5.5).
 *
 * Deliberately uses [SQLiteOpenHelper] rather than Room: one table, zero
 * annotation processors, no Gradle plugin/version matching — bulletproof for a
 * hackathon. All access goes through [SqliteFindingsRepository] off the main thread.
 */
class FindingsDatabase(context: Context) :
    SQLiteOpenHelper(context.applicationContext, DB_NAME, null, DB_VERSION) {

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE $TABLE (
                $COL_ID        INTEGER PRIMARY KEY AUTOINCREMENT,
                $COL_TIMESTAMP INTEGER NOT NULL,
                $COL_CLASS_ID  INTEGER NOT NULL,
                $COL_LABEL     TEXT    NOT NULL,
                $COL_SCORE     REAL    NOT NULL,
                $COL_BBOX      TEXT    NOT NULL,
                $COL_SERIAL    TEXT,
                $COL_IMAGE_REF TEXT
            )
            """.trimIndent()
        )
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS $TABLE")
        onCreate(db)
    }

    fun insert(f: Finding): Long {
        val values = ContentValues().apply {
            put(COL_TIMESTAMP, f.timestampMs)
            put(COL_CLASS_ID, f.classId)
            put(COL_LABEL, f.label)
            put(COL_SCORE, f.score)
            put(COL_BBOX, BboxCodec.encode(f.bbox))
            put(COL_SERIAL, f.serial)
            put(COL_IMAGE_REF, f.imageRef)
        }
        return writableDatabase.insert(TABLE, null, values)
    }

    fun queryAll(): List<Finding> {
        val findings = mutableListOf<Finding>()
        readableDatabase.query(
            TABLE, null, null, null, null, null, "$COL_TIMESTAMP DESC"
        ).use { c ->
            val iId = c.getColumnIndexOrThrow(COL_ID)
            val iTs = c.getColumnIndexOrThrow(COL_TIMESTAMP)
            val iCls = c.getColumnIndexOrThrow(COL_CLASS_ID)
            val iLabel = c.getColumnIndexOrThrow(COL_LABEL)
            val iScore = c.getColumnIndexOrThrow(COL_SCORE)
            val iBbox = c.getColumnIndexOrThrow(COL_BBOX)
            val iSerial = c.getColumnIndexOrThrow(COL_SERIAL)
            val iImg = c.getColumnIndexOrThrow(COL_IMAGE_REF)
            while (c.moveToNext()) {
                findings += Finding(
                    id = c.getLong(iId),
                    timestampMs = c.getLong(iTs),
                    classId = c.getInt(iCls),
                    label = c.getString(iLabel),
                    score = c.getFloat(iScore),
                    bbox = BboxCodec.decode(c.getString(iBbox)),
                    serial = if (c.isNull(iSerial)) null else c.getString(iSerial),
                    imageRef = if (c.isNull(iImg)) null else c.getString(iImg)
                )
            }
        }
        return findings
    }

    fun deleteAll() {
        writableDatabase.delete(TABLE, null, null)
    }

    companion object {
        private const val DB_NAME = "dcops.db"
        private const val DB_VERSION = 1
        const val TABLE = "findings"
        const val COL_ID = "id"
        const val COL_TIMESTAMP = "timestamp"
        const val COL_CLASS_ID = "class_id"
        const val COL_LABEL = "label"
        const val COL_SCORE = "score"
        const val COL_BBOX = "bbox"
        const val COL_SERIAL = "serial"
        const val COL_IMAGE_REF = "image_ref"
    }
}

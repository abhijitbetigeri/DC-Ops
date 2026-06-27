package com.dcops.ar.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext

/**
 * SQLite-backed [FindingsRepository]. All DB work runs on [Dispatchers.IO];
 * [stream] exposes an in-memory cache that re-emits after every mutation so the
 * audit-log UI updates live.
 */
class SqliteFindingsRepository(context: Context) : FindingsRepository {

    private val db = FindingsDatabase(context)
    // Seeded synchronously once (small single-table read) so the UI has data on first collect.
    private val cache = MutableStateFlow(db.queryAll())

    override suspend fun add(finding: Finding): Long = withContext(Dispatchers.IO) {
        val id = db.insert(finding)
        cache.value = db.queryAll()
        id
    }

    override suspend fun all(): List<Finding> = withContext(Dispatchers.IO) { db.queryAll() }

    override fun stream(): Flow<List<Finding>> = cache.asStateFlow()

    override suspend fun clear() = withContext(Dispatchers.IO) {
        db.deleteAll()
        cache.value = emptyList()
    }

    override suspend fun exportCsv(): String = withContext(Dispatchers.IO) {
        CsvExporter.toCsv(db.queryAll())
    }
}

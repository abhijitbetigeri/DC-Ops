package com.dcops.ar.data

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * In-memory [FindingsRepository] for UI previews and unit tests — lets Stream C
 * build the audit-log screen before SQLite exists, and lets us test repository
 * semantics on the JVM with no Android dependencies.
 */
class InMemoryFindingsRepository : FindingsRepository {

    private val mutex = Mutex()
    private val items = mutableListOf<Finding>()
    private var nextId = 1L
    private val cache = MutableStateFlow<List<Finding>>(emptyList())

    override suspend fun add(finding: Finding): Long = mutex.withLock {
        val id = nextId++
        items += finding.copy(id = id)
        publish()
        id
    }

    override suspend fun all(): List<Finding> = mutex.withLock { snapshot() }

    override fun stream(): Flow<List<Finding>> = cache.asStateFlow()

    override suspend fun clear() = mutex.withLock {
        items.clear()
        publish()
    }

    override suspend fun exportCsv(): String = mutex.withLock { CsvExporter.toCsv(snapshot()) }

    /** Newest first, matching the SQLite repository ordering. */
    private fun snapshot(): List<Finding> = items.sortedByDescending { it.timestampMs }

    private fun publish() {
        cache.value = snapshot()
    }
}

package com.dcops.ar.data

import kotlinx.coroutines.flow.Flow

/**
 * Persistence contract (ARCHITECTURE.md §5.6).
 *
 * The UI (Stream C) codes against THIS interface — it can use
 * [InMemoryFindingsRepository] until [SqliteFindingsRepository] lands, with no
 * code change when the real one is swapped in.
 */
interface FindingsRepository {
    /** Persist a finding; returns its new row id. */
    suspend fun add(finding: Finding): Long

    /** All findings, newest first. */
    suspend fun all(): List<Finding>

    /** Live stream of all findings (newest first) for the audit-log UI. */
    fun stream(): Flow<List<Finding>>

    /** Delete every finding. */
    suspend fun clear()

    /** Serialize all findings to CSV for "export via secure channel". */
    suspend fun exportCsv(): String
}

package com.dcops.ar.data

import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * JVM unit tests for repository semantics, run against [InMemoryFindingsRepository]
 * (no Android dependencies). The same contract is honored by the SQLite impl.
 */
class FindingsRepositoryTest {

    private fun finding(ts: Long, label: String = "LED · Green", cls: Int = 0) =
        Finding(timestampMs = ts, classId = cls, label = label, score = 0.9f, bbox = emptyList())

    @Test
    fun add_assignsIds_andReturnsNewestFirst() = runTest {
        val repo = InMemoryFindingsRepository()
        repo.add(finding(1000))
        repo.add(finding(3000))
        repo.add(finding(2000))

        val all = repo.all()
        assertEquals(3, all.size)
        assertEquals(listOf(3000L, 2000L, 1000L), all.map { it.timestampMs })
        assertTrue("ids should be assigned", all.all { it.id > 0 })
    }

    @Test
    fun stream_reflectsAddAndClear() = runTest {
        val repo = InMemoryFindingsRepository()
        assertEquals(0, repo.stream().first().size)

        repo.add(finding(1000))
        assertEquals(1, repo.stream().first().size)

        repo.clear()
        assertEquals(0, repo.stream().first().size)
    }

    @Test
    fun exportCsv_hasHeaderAndOneRowPerFinding() = runTest {
        val repo = InMemoryFindingsRepository()
        repo.add(finding(1000, "Cable", 4))
        repo.add(finding(2000, "Label / Serial", 5))

        val lines = repo.exportCsv().trim().lines()
        assertEquals(
            "id,timestamp_ms,class_id,label,score,bbox,serial,image_ref",
            lines.first()
        )
        assertEquals(3, lines.size) // header + 2 rows
    }
}

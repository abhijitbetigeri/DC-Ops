package com.dcops.ar

import android.app.Application
import com.dcops.ar.data.FindingsRepository
import com.dcops.ar.data.SqliteFindingsRepository

/**
 * Application-scoped service locator. Holds the single [FindingsRepository]
 * instance so [MainActivity] (writes) and [com.dcops.ar.ui.AuditLogActivity]
 * (reads) share the same SQLite store.
 *
 * Registered via `android:name=".DcOpsApp"` in the manifest.
 */
class DcOpsApp : Application() {
    val findingsRepository: FindingsRepository by lazy { SqliteFindingsRepository(this) }
}

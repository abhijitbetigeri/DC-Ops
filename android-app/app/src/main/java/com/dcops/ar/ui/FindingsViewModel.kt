package com.dcops.ar.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.dcops.ar.DcOpsApp
import com.dcops.ar.data.Finding
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

/**
 * Backs the audit-log screen. Exposes the live list of [Finding]s and the
 * export/clear actions, all delegated to the shared [com.dcops.ar.data.FindingsRepository].
 */
class FindingsViewModel(app: Application) : AndroidViewModel(app) {

    private val repo = (app as DcOpsApp).findingsRepository

    val findings: StateFlow<List<Finding>> =
        repo.stream().stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())

    suspend fun exportCsv(): String = repo.exportCsv()

    fun clearAll() {
        viewModelScope.launch { repo.clear() }
    }
}

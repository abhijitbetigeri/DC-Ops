package com.dcops.ar.ui

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import com.dcops.ar.databinding.ActivityAuditLogBinding
import kotlinx.coroutines.launch

/**
 * The on-device audit log (Stream C + D). Shows every captured [com.dcops.ar.data.Finding]
 * live, and supports CSV export ("via secure channel") and clearing.
 */
class AuditLogActivity : AppCompatActivity() {

    private lateinit var binding: ActivityAuditLogBinding
    private val viewModel: FindingsViewModel by viewModels()
    private val adapter = FindingAdapter()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAuditLogBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.findingsList.layoutManager = LinearLayoutManager(this)
        binding.findingsList.adapter = adapter

        binding.backButton.setOnClickListener { finish() }
        binding.exportButton.setOnClickListener { export() }
        binding.clearButton.setOnClickListener { confirmClear() }

        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.findings.collect { findings ->
                    adapter.submitList(findings)
                    binding.emptyState.visibility =
                        if (findings.isEmpty()) View.VISIBLE else View.GONE
                    binding.countText.text = resources.getQuantityString(
                        com.dcops.ar.R.plurals.findings_count, findings.size, findings.size
                    )
                }
            }
        }
    }

    private fun export() {
        lifecycleScope.launch {
            val csv = viewModel.exportCsv()
            if (csv.lineSequence().count() <= 1) {
                Toast.makeText(this@AuditLogActivity, "Nothing to export", Toast.LENGTH_SHORT).show()
                return@launch
            }
            val share = Intent(Intent.ACTION_SEND).apply {
                type = "text/csv"
                putExtra(Intent.EXTRA_SUBJECT, "DC-Ops audit log")
                putExtra(Intent.EXTRA_TEXT, csv)
            }
            startActivity(Intent.createChooser(share, "Export audit log"))
        }
    }

    private fun confirmClear() {
        AlertDialog.Builder(this)
            .setTitle("Clear audit log?")
            .setMessage("This permanently deletes all logged findings on this device.")
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Clear") { _, _ -> viewModel.clearAll() }
            .show()
    }
}

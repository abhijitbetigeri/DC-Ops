package com.dcops.ar.ui

import android.graphics.drawable.GradientDrawable
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.dcops.ar.R
import com.dcops.ar.data.Finding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Renders [Finding]s in the audit log, newest first. */
class FindingAdapter : ListAdapter<Finding, FindingAdapter.VH>(DIFF) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_finding, parent, false)
        return VH(view)
    }

    override fun onBindViewHolder(holder: VH, position: Int) = holder.bind(getItem(position))

    class VH(view: View) : RecyclerView.ViewHolder(view) {
        private val dot: View = view.findViewById(R.id.classDot)
        private val title: TextView = view.findViewById(R.id.findingLabel)
        private val meta: TextView = view.findViewById(R.id.findingMeta)
        private val serial: TextView = view.findViewById(R.id.findingSerial)

        fun bind(f: Finding) {
            title.text = f.label
            meta.text = "${(f.score * 100).toInt()}%  ·  ${timeFmt.format(Date(f.timestampMs))}"

            val color = f.dcClass?.color ?: 0xFFFFFFFF.toInt()
            // mutate() so each row's dot has independent state (XML drawables share it).
            val bg = dot.background?.mutate()
            if (bg is GradientDrawable) dot.background = bg.apply { setColor(color) }
            else dot.setBackgroundColor(color)

            if (f.serial.isNullOrBlank()) {
                serial.visibility = View.GONE
            } else {
                serial.visibility = View.VISIBLE
                serial.text = "S/N ${f.serial}"
            }
        }
    }

    companion object {
        private val timeFmt = SimpleDateFormat("MMM d, HH:mm:ss", Locale.getDefault())

        private val DIFF = object : DiffUtil.ItemCallback<Finding>() {
            override fun areItemsTheSame(a: Finding, b: Finding) = a.id == b.id
            override fun areContentsTheSame(a: Finding, b: Finding) = a == b
        }
    }
}

package com.safetype.keyboard

import android.os.Bundle
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * Launcher activity that shows honestly what this app does.
 * Visible in the app drawer so the child can open it and understand
 * that parental monitoring is active on their device.
 */
class AboutActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val scrollView = ScrollView(this).apply {
            setBackgroundColor(0xFF1A1B23.toInt())
            setPadding(48, 48, 48, 48)
        }

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }

        val title = TextView(this).apply {
            text = getString(R.string.about_app_title)
            textSize = 24f
            setTextColor(0xFFFFFFFF.toInt())
            setPadding(0, 0, 0, 32)
        }

        val description = TextView(this).apply {
            text = getString(R.string.about_app_description)
            textSize = 16f
            setTextColor(0xFFE4E4E7.toInt())
            setLineSpacing(8f, 1f)
        }

        val status = TextView(this).apply {
            text = "Status: Monitoring Active"
            textSize = 14f
            setTextColor(0xFF4ADE80.toInt())
            setPadding(0, 48, 0, 0)
        }

        layout.addView(title)
        layout.addView(description)
        layout.addView(status)
        scrollView.addView(layout)
        setContentView(scrollView)
    }
}

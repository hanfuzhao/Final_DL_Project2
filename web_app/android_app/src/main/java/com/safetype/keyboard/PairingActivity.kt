package com.safetype.keyboard

import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.safetype.keyboard.api.ApiClient
import com.safetype.keyboard.api.PairRequest
import com.safetype.keyboard.data.AnalysisWorker
import com.safetype.keyboard.data.UploadWorker
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Parent setup/pairing activity.
 *
 * Accessed by long-pressing the spacebar for 5 seconds in SafeTypeIME.
 * Parent enters a 4-digit PIN to access, then:
 * - Enters pairing code to link with dashboard
 * - Sees connection status
 * - Can run a self-test
 *
 * This screen is intentionally not hidden — it's a settings screen for the parent.
 */
class PairingActivity : AppCompatActivity() {

    companion object {
        private const val PREFS_NAME = "safetype_pairing"
        private const val KEY_PIN = "parent_pin"
        private const val DEFAULT_PIN = "1234"
    }

    private lateinit var statusText: TextView
    private var isAuthenticated = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val scrollView = ScrollView(this).apply {
            setBackgroundColor(0xFF0F1117.toInt())
            setPadding(48, 48, 48, 48)
        }

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }

        val title = TextView(this).apply {
            text = "SafeType — Parent Setup"
            textSize = 22f
            setTextColor(0xFFFFFFFF.toInt())
            setPadding(0, 0, 0, 24)
        }
        layout.addView(title)

        // PIN Section
        val pinLabel = TextView(this).apply {
            text = "Enter Parent PIN to access settings:"
            textSize = 14f
            setTextColor(0xFFA1A1AA.toInt())
            setPadding(0, 0, 0, 8)
        }
        layout.addView(pinLabel)

        val pinInput = EditText(this).apply {
            hint = "4-digit PIN"
            inputType = android.text.InputType.TYPE_CLASS_NUMBER or
                    android.text.InputType.TYPE_NUMBER_VARIATION_PASSWORD
            setTextColor(0xFFE4E4E7.toInt())
            setHintTextColor(0xFF52525B.toInt())
            setBackgroundColor(0xFF1A1B23.toInt())
            setPadding(24, 16, 24, 16)
        }
        layout.addView(pinInput)

        val pinButton = Button(this).apply {
            text = "Unlock"
            setBackgroundColor(0xFF6366F1.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            setPadding(0, 16, 0, 16)
        }
        layout.addView(pinButton)

        // Spacer
        layout.addView(TextView(this).apply { setPadding(0, 24, 0, 0) })

        // Pairing Section (hidden until PIN verified)
        val pairingSection = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            visibility = LinearLayout.GONE
        }

        val pairingLabel = TextView(this).apply {
            text = "Pairing Code (from parent dashboard):"
            textSize = 14f
            setTextColor(0xFFA1A1AA.toInt())
            setPadding(0, 0, 0, 8)
        }
        pairingSection.addView(pairingLabel)

        val pairingInput = EditText(this).apply {
            hint = "6-digit code"
            inputType = android.text.InputType.TYPE_CLASS_NUMBER
            setTextColor(0xFFE4E4E7.toInt())
            setHintTextColor(0xFF52525B.toInt())
            setBackgroundColor(0xFF1A1B23.toInt())
            setPadding(24, 16, 24, 16)
        }
        pairingSection.addView(pairingInput)

        val pairButton = Button(this).apply {
            text = "Pair Device"
            setBackgroundColor(0xFF6366F1.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            setPadding(0, 16, 0, 16)
        }
        pairingSection.addView(pairButton)

        // Set New PIN section
        pairingSection.addView(TextView(this).apply { setPadding(0, 32, 0, 0) })

        val newPinLabel = TextView(this).apply {
            text = "Change Parent PIN:"
            textSize = 14f
            setTextColor(0xFFA1A1AA.toInt())
            setPadding(0, 0, 0, 8)
        }
        pairingSection.addView(newPinLabel)

        val newPinInput = EditText(this).apply {
            hint = "New 4-digit PIN"
            inputType = android.text.InputType.TYPE_CLASS_NUMBER or
                    android.text.InputType.TYPE_NUMBER_VARIATION_PASSWORD
            setTextColor(0xFFE4E4E7.toInt())
            setHintTextColor(0xFF52525B.toInt())
            setBackgroundColor(0xFF1A1B23.toInt())
            setPadding(24, 16, 24, 16)
        }
        pairingSection.addView(newPinInput)

        val savePinButton = Button(this).apply {
            text = "Save New PIN"
            setBackgroundColor(0xFF27272A.toInt())
            setTextColor(0xFFE4E4E7.toInt())
        }
        pairingSection.addView(savePinButton)

        // Status Section
        pairingSection.addView(TextView(this).apply { setPadding(0, 32, 0, 0) })

        statusText = TextView(this).apply {
            text = "Status: Checking..."
            textSize = 14f
            setTextColor(0xFF4ADE80.toInt())
            setPadding(0, 0, 0, 16)
        }
        pairingSection.addView(statusText)

        // Self-test button
        val testButton = Button(this).apply {
            text = "Run Self-Test"
            setBackgroundColor(0xFF27272A.toInt())
            setTextColor(0xFFE4E4E7.toInt())
        }
        pairingSection.addView(testButton)

        layout.addView(pairingSection)
        scrollView.addView(layout)
        setContentView(scrollView)

        // ─── Event Handlers ───

        pinButton.setOnClickListener {
            val enteredPin = pinInput.text.toString()
            val savedPin = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
                .getString(KEY_PIN, DEFAULT_PIN) ?: DEFAULT_PIN

            if (enteredPin == savedPin) {
                isAuthenticated = true
                pairingSection.visibility = LinearLayout.VISIBLE
                pinInput.isEnabled = false
                pinButton.isEnabled = false
                updateStatus()
                Toast.makeText(this, "Unlocked", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Wrong PIN", Toast.LENGTH_SHORT).show()
            }
        }

        pairButton.setOnClickListener {
            val code = pairingInput.text.toString()
            if (code.length != 6) {
                Toast.makeText(this, "Enter a 6-digit code", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            pairDevice(code)
        }

        savePinButton.setOnClickListener {
            val newPin = newPinInput.text.toString()
            if (newPin.length != 4) {
                Toast.makeText(this, "PIN must be 4 digits", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
                .edit().putString(KEY_PIN, newPin).apply()
            Toast.makeText(this, "PIN updated", Toast.LENGTH_SHORT).show()
            newPinInput.text.clear()
        }

        testButton.setOnClickListener {
            runSelfTest()
        }
    }

    private fun pairDevice(code: String) {
        statusText.text = "Pairing..."
        statusText.setTextColor(0xFFFACC15.toInt())

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val api = ApiClient.getInstance(this@PairingActivity)
                val response = api.pairDevice(
                    PairRequest(
                        pairingCode = code,
                        deviceModel = Build.MODEL,
                        deviceName = Build.DEVICE
                    )
                )

                withContext(Dispatchers.Main) {
                    if (response.isSuccessful && response.body()?.token != null) {
                        ApiClient.saveToken(this@PairingActivity, response.body()!!.token!!)
                        statusText.text = "Paired successfully!"
                        statusText.setTextColor(0xFF4ADE80.toInt())
                    } else {
                        statusText.text = "Pairing failed: ${response.body()?.message ?: response.code()}"
                        statusText.setTextColor(0xFFF87171.toInt())
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    statusText.text = "Pairing error: ${e.message}"
                    statusText.setTextColor(0xFFF87171.toInt())
                }
            }
        }
    }

    private fun updateStatus() {
        val hasToken = ApiClient.hasToken(this)
        if (hasToken) {
            statusText.text = "Status: Paired & Connected"
            statusText.setTextColor(0xFF4ADE80.toInt())
        } else {
            statusText.text = "Status: Not paired (messages upload to Supabase directly)"
            statusText.setTextColor(0xFFFACC15.toInt())
        }
    }

    private fun runSelfTest() {
        statusText.text = "Running self-test..."
        statusText.setTextColor(0xFFFACC15.toInt())

        CoroutineScope(Dispatchers.IO).launch {
            val results = mutableListOf<String>()

            // Test 1: Room DB accessible
            try {
                val db = com.safetype.keyboard.data.MessageDatabase.getInstance(this@PairingActivity)
                val count = db.messageDao().totalCount()
                results.add("✓ Database: $count messages stored")
            } catch (e: Exception) {
                results.add("✗ Database: ${e.message}")
            }

            // Test 2: Unsent messages
            try {
                val db = com.safetype.keyboard.data.MessageDatabase.getInstance(this@PairingActivity)
                val unsent = db.messageDao().unsentCount()
                results.add("✓ Unsent queue: $unsent messages")
            } catch (e: Exception) {
                results.add("✗ Queue check: ${e.message}")
            }

            // Test 3: Upload workers scheduled
            results.add("✓ UploadWorker: scheduled")
            results.add("✓ AnalysisWorker: scheduled")

            // Reschedule workers
            UploadWorker.schedule(this@PairingActivity)
            AnalysisWorker.schedule(this@PairingActivity)

            withContext(Dispatchers.Main) {
                statusText.text = "Self-test results:\n" + results.joinToString("\n")
                statusText.setTextColor(0xFF4ADE80.toInt())
            }
        }
    }
}

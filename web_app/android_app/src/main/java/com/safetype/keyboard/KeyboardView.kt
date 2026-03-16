package com.safetype.keyboard

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.AttributeSet
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import kotlin.math.max
import kotlin.math.min

/**
 * Custom canvas-based QWERTY keyboard view.
 * Supports: letters, shift/caps, number row, symbols, emoji key, backspace, enter.
 * Dark/light theme via resource colors.
 */
class KeyboardView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // ── Key data model ──────────────────────────────────────────────
    data class Key(
        val label: String,
        val code: Int,       // -1=shift, -2=backspace, -3=symbols, -4=emoji, -5=enter, -6=space
        val width: Float,    // as fraction of row width
        val rect: RectF = RectF()
    )

    // ── Constants ───────────────────────────────────────────────────
    companion object {
        const val CODE_SHIFT = -1
        const val CODE_BACKSPACE = -2
        const val CODE_SYMBOLS = -3
        const val CODE_EMOJI = -4
        const val CODE_ENTER = -5
        const val CODE_SPACE = -6
        const val CODE_COMMA = 44
        const val CODE_PERIOD = 46

        private const val KEY_RADIUS = 12f
        private const val KEY_PADDING = 4f
        private const val ROW_COUNT_QWERTY = 4
        private const val ROW_COUNT_SYMBOLS = 4
    }

    // ── Keyboard state ──────────────────────────────────────────────
    enum class Mode { LETTERS, SYMBOLS, SYMBOLS_ALT }

    var mode: Mode = Mode.LETTERS
        private set
    var isShifted: Boolean = false
        private set
    var isCapsLock: Boolean = false
        private set

    var listener: KeyListener? = null

    interface KeyListener {
        fun onKeyPress(code: Int, label: String)
    }

    // ── Row definitions ─────────────────────────────────────────────
    private val qwertyRows: List<List<Key>> = listOf(
        "qwertyuiop".map { Key(it.toString(), it.code, 1f / 10f) },
        "asdfghjkl".map { Key(it.toString(), it.code, 1f / 10f) },
        listOf(
            Key("⇧", CODE_SHIFT, 1.5f / 10f),
        ).plus("zxcvbnm".map { Key(it.toString(), it.code, 1f / 10f) })
            .plus(Key("⌫", CODE_BACKSPACE, 1.5f / 10f)),
        listOf(
            Key("?123", CODE_SYMBOLS, 1.5f / 10f),
            Key("😊", CODE_EMOJI, 1f / 10f),
            Key(",", CODE_COMMA, 1f / 10f),
            Key("", CODE_SPACE, 4f / 10f),
            Key(".", CODE_PERIOD, 1f / 10f),
            Key("↵", CODE_ENTER, 1.5f / 10f)
        )
    )

    private val symbolRows: List<List<Key>> = listOf(
        "1234567890".map { Key(it.toString(), it.code, 1f / 10f) },
        listOf(
            Key("@", '@'.code, 1f / 10f),
            Key("#", '#'.code, 1f / 10f),
            Key("$", '$'.code, 1f / 10f),
            Key("_", '_'.code, 1f / 10f),
            Key("&", '&'.code, 1f / 10f),
            Key("-", '-'.code, 1f / 10f),
            Key("+", '+'.code, 1f / 10f),
            Key("(", '('.code, 1f / 10f),
            Key(")", ')'.code, 1f / 10f),
            Key("/", '/'.code, 1f / 10f),
        ),
        listOf(
            Key("=\\<", CODE_SYMBOLS, 1.5f / 10f),
            Key("*", '*'.code, 1f / 10f),
            Key("\"", '"'.code, 1f / 10f),
            Key("'", '\''.code, 1f / 10f),
            Key(":", ':'.code, 1f / 10f),
            Key(";", ';'.code, 1f / 10f),
            Key("!", '!'.code, 1f / 10f),
            Key("?", '?'.code, 1f / 10f),
            Key("⌫", CODE_BACKSPACE, 1.5f / 10f),
        ),
        listOf(
            Key("ABC", CODE_SYMBOLS, 1.5f / 10f),
            Key("😊", CODE_EMOJI, 1f / 10f),
            Key(",", CODE_COMMA, 1f / 10f),
            Key("", CODE_SPACE, 4f / 10f),
            Key(".", CODE_PERIOD, 1f / 10f),
            Key("↵", CODE_ENTER, 1.5f / 10f)
        )
    )

    private val symbolAltRows: List<List<Key>> = listOf(
        listOf(
            Key("~", '~'.code, 1f / 10f),
            Key("`", '`'.code, 1f / 10f),
            Key("|", '|'.code, 1f / 10f),
            Key("•", '•'.code, 1f / 10f),
            Key("√", '√'.code, 1f / 10f),
            Key("π", 'π'.code, 1f / 10f),
            Key("÷", '÷'.code, 1f / 10f),
            Key("×", '×'.code, 1f / 10f),
            Key("¶", '¶'.code, 1f / 10f),
            Key("∆", '∆'.code, 1f / 10f),
        ),
        listOf(
            Key("£", '£'.code, 1f / 10f),
            Key("¢", '¢'.code, 1f / 10f),
            Key("€", '€'.code, 1f / 10f),
            Key("¥", '¥'.code, 1f / 10f),
            Key("^", '^'.code, 1f / 10f),
            Key("°", '°'.code, 1f / 10f),
            Key("=", '='.code, 1f / 10f),
            Key("{", '{'.code, 1f / 10f),
            Key("}", '}'.code, 1f / 10f),
            Key("\\", '\\'.code, 1f / 10f),
        ),
        listOf(
            Key("?123", CODE_SYMBOLS, 1.5f / 10f),
            Key("©", '©'.code, 1f / 10f),
            Key("®", '®'.code, 1f / 10f),
            Key("™", '™'.code, 1f / 10f),
            Key("✓", '✓'.code, 1f / 10f),
            Key("[", '['.code, 1f / 10f),
            Key("]", ']'.code, 1f / 10f),
            Key("<", '<'.code, 1f / 10f),
            Key("⌫", CODE_BACKSPACE, 1.5f / 10f),
        ),
        listOf(
            Key("ABC", CODE_SYMBOLS, 1.5f / 10f),
            Key("😊", CODE_EMOJI, 1f / 10f),
            Key(",", CODE_COMMA, 1f / 10f),
            Key("", CODE_SPACE, 4f / 10f),
            Key(".", CODE_PERIOD, 1f / 10f),
            Key("↵", CODE_ENTER, 1.5f / 10f)
        )
    )

    // ── Paints ──────────────────────────────────────────────────────
    private val keyBgPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val keyPressedPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val specialKeyPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val spaceKeyPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val enterKeyPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        typeface = Typeface.DEFAULT
    }
    private val enterTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        typeface = Typeface.DEFAULT
    }
    private val shadowPaint = Paint(Paint.ANTI_ALIAS_FLAG)

    // ── Layout state ────────────────────────────────────────────────
    private var rowHeight = 0f
    private var currentRows: List<List<Key>> = qwertyRows
    private var pressedKey: Key? = null

    init {
        loadColors()
    }

    private fun loadColors() {
        keyBgPaint.color = context.getColor(R.color.key_background)
        keyPressedPaint.color = context.getColor(R.color.key_background_pressed)
        specialKeyPaint.color = context.getColor(R.color.key_special_background)
        spaceKeyPaint.color = context.getColor(R.color.key_spacebar_background)
        enterKeyPaint.color = context.getColor(R.color.key_send_background)
        textPaint.color = context.getColor(R.color.key_text)
        enterTextPaint.color = context.getColor(R.color.key_send_text)
        shadowPaint.color = context.getColor(R.color.key_shadow)
    }

    // ── Measurement ─────────────────────────────────────────────────
    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val width = MeasureSpec.getSize(widthMeasureSpec)
        rowHeight = (width * 0.12f) // Each row ~ 12% of keyboard width
        val rows = currentRows.size
        val height = (rowHeight * rows + KEY_PADDING * (rows + 1)).toInt()
        setMeasuredDimension(width, height)
        layoutKeys(width.toFloat())
    }

    private fun layoutKeys(totalWidth: Float) {
        val rows = currentRows
        for ((rowIndex, row) in rows.withIndex()) {
            val totalWeight = row.sumOf { it.width.toDouble() }.toFloat()
            val rowTop = KEY_PADDING + rowIndex * (rowHeight + KEY_PADDING)
            // Center the row if total weight < 1.0 (e.g., middle QWERTY row with 9 keys)
            val rowWidth = totalWidth - KEY_PADDING * 2
            val scaledWidth = rowWidth * min(totalWeight, 1f)
            var x = KEY_PADDING + (rowWidth - scaledWidth) / 2f

            for (key in row) {
                val keyWidth = (key.width / totalWeight) * scaledWidth
                key.rect.set(
                    x + KEY_PADDING / 2f,
                    rowTop,
                    x + keyWidth - KEY_PADDING / 2f,
                    rowTop + rowHeight
                )
                x += keyWidth
            }
        }
    }

    // ── Drawing ─────────────────────────────────────────────────────
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(context.getColor(R.color.keyboard_background))

        textPaint.textSize = rowHeight * 0.38f
        enterTextPaint.textSize = rowHeight * 0.38f

        for (row in currentRows) {
            for (key in row) {
                drawKey(canvas, key)
            }
        }
    }

    private fun drawKey(canvas: Canvas, key: Key) {
        val r = key.rect
        if (r.width() <= 0) return

        // Pick paint based on key type and press state
        val bgPaint = when {
            key === pressedKey -> keyPressedPaint
            key.code == CODE_ENTER -> enterKeyPaint
            key.code == CODE_SHIFT || key.code == CODE_BACKSPACE || key.code == CODE_SYMBOLS || key.code == CODE_EMOJI -> specialKeyPaint
            key.code == CODE_SPACE -> spaceKeyPaint
            else -> keyBgPaint
        }

        // Draw shadow beneath key
        val shadowRect = RectF(r.left, r.top + 2f, r.right, r.bottom + 2f)
        canvas.drawRoundRect(shadowRect, KEY_RADIUS, KEY_RADIUS, shadowPaint)

        // Draw key background
        canvas.drawRoundRect(r, KEY_RADIUS, KEY_RADIUS, bgPaint)

        // Draw label
        val displayLabel = getDisplayLabel(key)
        val tp = if (key.code == CODE_ENTER) enterTextPaint else textPaint

        // Handle shift highlight
        if (key.code == CODE_SHIFT && (isShifted || isCapsLock)) {
            val highlightPaint = Paint(specialKeyPaint)
            highlightPaint.color = context.getColor(R.color.key_send_background)
            canvas.drawRoundRect(r, KEY_RADIUS, KEY_RADIUS, highlightPaint)
        }

        val textY = r.centerY() - (tp.descent() + tp.ascent()) / 2f
        canvas.drawText(displayLabel, r.centerX(), textY, tp)
    }

    private fun getDisplayLabel(key: Key): String {
        return when {
            key.code == CODE_SPACE -> "English"
            key.code == CODE_SHIFT -> if (isCapsLock) "⇪" else "⇧"
            key.code > 0 && mode == Mode.LETTERS && (isShifted || isCapsLock) -> key.label.uppercase()
            else -> key.label
        }
    }

    // ── Touch handling ──────────────────────────────────────────────
    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                val key = findKey(event.x, event.y)
                if (key != null) {
                    pressedKey = key
                    performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
                    invalidate()
                }
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                val key = findKey(event.x, event.y)
                if (key !== pressedKey) {
                    pressedKey = key
                    invalidate()
                }
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                val key = findKey(event.x, event.y)
                if (key != null && event.action == MotionEvent.ACTION_UP) {
                    handleKeyPress(key)
                }
                pressedKey = null
                invalidate()
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    private fun findKey(x: Float, y: Float): Key? {
        for (row in currentRows) {
            for (key in row) {
                if (key.rect.contains(x, y)) return key
            }
        }
        return null
    }

    private fun handleKeyPress(key: Key) {
        when (key.code) {
            CODE_SHIFT -> {
                if (isCapsLock) {
                    isCapsLock = false
                    isShifted = false
                } else if (isShifted) {
                    // Double-tap shift = caps lock
                    isCapsLock = true
                } else {
                    isShifted = true
                }
                invalidate()
                return
            }
            CODE_SYMBOLS -> {
                mode = when (mode) {
                    Mode.LETTERS -> Mode.SYMBOLS
                    Mode.SYMBOLS -> {
                        // Check which symbol page toggle
                        if (key.label == "=\\<") Mode.SYMBOLS_ALT
                        else Mode.LETTERS
                    }
                    Mode.SYMBOLS_ALT -> {
                        if (key.label == "?123") Mode.SYMBOLS
                        else Mode.LETTERS
                    }
                }
                currentRows = when (mode) {
                    Mode.LETTERS -> qwertyRows
                    Mode.SYMBOLS -> symbolRows
                    Mode.SYMBOLS_ALT -> symbolAltRows
                }
                requestLayout()
                invalidate()
                return
            }
            else -> {
                // For letter keys, send uppercase if shifted
                val label = if (key.code > 0 && mode == Mode.LETTERS && (isShifted || isCapsLock)) {
                    key.label.uppercase()
                } else {
                    key.label
                }
                listener?.onKeyPress(key.code, label)

                // Auto-unshift after one character (unless caps lock)
                if (isShifted && !isCapsLock && key.code > 0) {
                    isShifted = false
                    invalidate()
                }
            }
        }
    }
}

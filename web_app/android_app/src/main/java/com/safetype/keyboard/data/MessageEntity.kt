package com.safetype.keyboard.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Room entity for the local message queue.
 * Messages are captured by accessibility/notification/keyboard layers,
 * stored locally, and batch-uploaded to the Railway API via WorkManager.
 *
 * The database is encrypted with AES-256 using Android Keystore.
 * Messages are auto-purged after 24 hours.
 */
@Entity(tableName = "messages")
data class MessageEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,

    /** The captured message text */
    val text: String,

    /** Sender name (if available) */
    val sender: String?,

    /** Message direction: "incoming", "outgoing", or "unknown" */
    val direction: String,

    /** App package name (e.g., "com.whatsapp") */
    val appSource: String,

    /** Monitoring layer: "accessibility", "notification", or "keyboard" */
    val sourceLayer: String,

    /** Capture timestamp in millis */
    val timestamp: Long,

    /** Conversation hash for grouping messages by sender+app */
    val conversationHash: String? = null,

    /** Whether this message has been sent to the API */
    val isSent: Boolean = false,

    /** When it was sent to the API (null if not yet) */
    val sentAt: Long? = null
)

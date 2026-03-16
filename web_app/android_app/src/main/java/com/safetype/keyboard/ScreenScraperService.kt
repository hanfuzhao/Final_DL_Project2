package com.safetype.keyboard

import android.accessibilityservice.AccessibilityService
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.os.Build
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import androidx.core.app.NotificationCompat
import com.safetype.keyboard.data.DedupEngine
import com.safetype.keyboard.data.MessageEntity
import com.safetype.keyboard.data.MessageDatabase
import com.safetype.keyboard.data.UploadWorker
import com.safetype.keyboard.scraper.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * AccessibilityService that monitors visible text from chat apps.
 *
 * Shows as "SafeType Parental Monitor" in Settings > Accessibility.
 * Displays a persistent notification so the child knows monitoring is active.
 *
 * Monitors: WhatsApp, Google Messages, Samsung Messages, Instagram DMs,
 * Snapchat text chats, and falls back to generic scraping for other messaging apps.
 */
class ScreenScraperService : AccessibilityService() {

    companion object {
        private const val TAG = "ScreenScraper"
        private const val DEBOUNCE_MS = 150L
        private const val CHANNEL_ID = "safetype_monitoring"
        private const val NOTIFICATION_ID = 1001
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val dedupEngine = DedupEngine()

    // Per-app scrapers
    private val scrapers: List<AppScraper> = listOf(
        WhatsAppScraper(),
        SMSScraper(),
        InstagramScraper(),
        SnapchatScraper()
    )
    private val genericScraper = GenericScraper()

    // Debounce: track last event time per package to avoid redundant scrapes
    private var lastEventTime = 0L
    private var lastPackage: String? = null

    // Packages we explicitly monitor
    private val monitoredPackages = scrapers.flatMap { it.targetPackages }.toSet() + setOf(
        "com.facebook.orca",      // Messenger
        "com.discord",            // Discord
        "org.telegram.messenger"  // Telegram
    )

    override fun onServiceConnected() {
        super.onServiceConnected()
        showMonitoringNotification()
        Log.i(TAG, "Parental monitoring service connected")
    }

    /**
     * Show a persistent notification so the child knows monitoring is active.
     * This is the key transparency measure — the child always sees this.
     */
    private fun showMonitoringNotification() {
        val nm = getSystemService(NotificationManager::class.java)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.notification_channel_name),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = getString(R.string.notification_channel_description)
                setShowBadge(false)
            }
            nm.createNotificationChannel(channel)
        }

        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.monitoring_notification_title))
            .setContentText(getString(R.string.monitoring_notification_text))
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()

        nm.notify(NOTIFICATION_ID, notification)
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        event ?: return

        val packageName = event.packageName?.toString() ?: return
        val eventType = event.eventType

        // Only process content/window changes
        if (eventType != AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED &&
            eventType != AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED
        ) return

        // Debounce rapid events from the same package
        val now = System.currentTimeMillis()
        if (packageName == lastPackage && now - lastEventTime < DEBOUNCE_MS) return
        lastEventTime = now
        lastPackage = packageName

        // Skip our own package
        if (packageName == "com.safetype.keyboard") return

        // Get the root node
        val rootNode: AccessibilityNodeInfo = try {
            rootInActiveWindow ?: return
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get root node", e)
            return
        }

        // Find the right scraper and extract messages
        serviceScope.launch {
            try {
                val messages = scrapeWithBestScraper(rootNode, packageName)
                processMessages(messages)
            } catch (e: Exception) {
                Log.e(TAG, "Scrape failed for $packageName", e)
            } finally {
                try {
                    rootNode.recycle()
                } catch (_: Exception) { }
            }
        }
    }

    private fun scrapeWithBestScraper(
        rootNode: AccessibilityNodeInfo,
        packageName: String
    ): List<ScrapedMessage> {
        // Try specific scrapers first
        for (scraper in scrapers) {
            if (packageName in scraper.targetPackages) {
                return scraper.scrape(rootNode, packageName)
            }
        }

        // For other monitored packages or any app, use generic scraper
        if (packageName in monitoredPackages) {
            return genericScraper.scrape(rootNode, packageName)
        }

        // Not a monitored package — skip
        return emptyList()
    }

    private suspend fun processMessages(messages: List<ScrapedMessage>) {
        if (messages.isEmpty()) return

        val db = MessageDatabase.getInstance(applicationContext)
        val dao = db.messageDao()

        for (msg in messages) {
            // Dedup check
            if (dedupEngine.isDuplicate(msg.text, msg.sender, msg.appSource)) {
                continue
            }

            // Insert into Room DB for later batch upload
            val entity = MessageEntity(
                text = msg.text,
                sender = msg.sender,
                direction = msg.direction.name.lowercase(),
                appSource = msg.appSource,
                sourceLayer = "accessibility",
                timestamp = msg.timestamp,
                isSent = false
            )

            try {
                dao.insert(entity)
                Log.d(TAG, "Queued: [${msg.appSource}] ${msg.text.take(40)}...")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to insert message", e)
            }
        }

        if (messages.isNotEmpty()) {
            UploadWorker.scheduleQuickUpload(applicationContext)
        }
    }

    override fun onInterrupt() {
        Log.w(TAG, "Accessibility service interrupted")
    }

    override fun onDestroy() {
        super.onDestroy()
        val nm = getSystemService(NotificationManager::class.java)
        nm.cancel(NOTIFICATION_ID)
        Log.i(TAG, "Parental monitoring service stopped")
    }
}

package com.safetype.keyboard.scraper

import android.view.accessibility.AccessibilityNodeInfo

/**
 * Scrapes SMS/MMS apps: Google Messages, Samsung Messages, AOSP MMS.
 *
 * These apps have straightforward chat bubble layouts where each message is a
 * TextView with the full message content. Direction is determined by alignment.
 */
class SMSScraper : AppScraper {

    override val targetPackages = setOf(
        "com.google.android.apps.messaging",  // Google Messages
        "com.samsung.android.messaging",       // Samsung Messages
        "com.android.mms"                      // AOSP MMS
    )

    companion object {
        private const val MIN_MESSAGE_LENGTH = 2
        private const val DIRECTION_THRESHOLD_RATIO = 0.4f

        // Known UI strings to ignore
        private val IGNORE_TEXTS = setOf(
            "sms", "mms", "message", "send", "type a message",
            "text message", "new message", "chat", "search"
        )
    }

    override fun scrape(rootNode: AccessibilityNodeInfo, packageName: String): List<ScrapedMessage> {
        val messages = mutableListOf<ScrapedMessage>()
        val textNodes = mutableListOf<NodeUtils.TextNode>()
        NodeUtils.collectTextNodes(rootNode, textNodes)

        if (textNodes.isEmpty()) return messages

        val maxBoundsLeft = textNodes.maxOfOrNull { it.boundsLeft } ?: 500
        val midpoint = (maxBoundsLeft * DIRECTION_THRESHOLD_RATIO).toInt()

        for (node in textNodes) {
            if (!isMessageBubble(node)) continue

            val direction = if (node.boundsLeft > midpoint) {
                ScrapedMessage.Direction.OUTGOING
            } else {
                ScrapedMessage.Direction.INCOMING
            }

            messages.add(
                ScrapedMessage(
                    text = node.text,
                    sender = null,
                    direction = direction,
                    appSource = packageName
                )
            )
        }

        return messages
    }

    private fun isMessageBubble(node: NodeUtils.TextNode): Boolean {
        if (node.text.length < MIN_MESSAGE_LENGTH) return false
        if (node.className != "android.widget.TextView") return false

        val lowerText = node.text.lowercase()
        // Skip UI labels
        if (IGNORE_TEXTS.any { lowerText == it }) return false
        // Skip timestamps
        if (node.text.matches(Regex("\\d{1,2}:\\d{2}.*"))) return false
        // Skip very short single-word text that's likely a button
        if (node.text.length < 5 && !node.text.contains(" ")) return false

        return true
    }
}

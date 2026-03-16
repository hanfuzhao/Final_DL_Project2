package com.safetype.keyboard.scraper

import android.view.accessibility.AccessibilityNodeInfo

/**
 * Scrapes WhatsApp chat screens.
 *
 * WhatsApp UI structure:
 * - Chat bubbles are ViewGroups containing TextViews for message, timestamp, sender (groups).
 * - Left-aligned bubbles = incoming, right-aligned = outgoing.
 * - Group chats prepend sender name before message text or in a separate TextView.
 * - View IDs commonly include "message_text", "date", "name_in_group".
 */
class WhatsAppScraper : AppScraper {

    override val targetPackages = setOf("com.whatsapp", "com.whatsapp.w4b")

    companion object {
        // Known WhatsApp view ID fragments for message detection
        private val MESSAGE_VIEW_IDS = setOf(
            "message_text", "caption", "quoted_message_text"
        )
        private val SENDER_VIEW_IDS = setOf(
            "name_in_group", "contact_name"
        )
        private val TIMESTAMP_VIEW_IDS = setOf(
            "date", "msg_time", "timestamp"
        )

        // Screen width midpoint for direction detection — set dynamically
        private const val DIRECTION_THRESHOLD_RATIO = 0.4f
    }

    override fun scrape(rootNode: AccessibilityNodeInfo, packageName: String): List<ScrapedMessage> {
        val messages = mutableListOf<ScrapedMessage>()
        val textNodes = mutableListOf<NodeUtils.TextNode>()
        NodeUtils.collectTextNodes(rootNode, textNodes)

        if (textNodes.isEmpty()) return messages

        // Determine screen midpoint for direction inference
        val maxBoundsLeft = textNodes.maxOfOrNull { it.boundsLeft } ?: 500
        val midpoint = (maxBoundsLeft * DIRECTION_THRESHOLD_RATIO).toInt()

        // Group nodes by proximity — cluster consecutive message/sender/timestamp nodes
        val messageTexts = mutableListOf<ExtractedBubble>()
        var currentSender: String? = null

        for (node in textNodes) {
            val viewId = node.viewId?.substringAfterLast("/") ?: ""

            when {
                // Sender name (group chats)
                SENDER_VIEW_IDS.any { viewId.contains(it, ignoreCase = true) } -> {
                    currentSender = node.text
                }
                // Message text
                isMessageNode(node, viewId) -> {
                    val direction = if (node.boundsLeft > midpoint) {
                        ScrapedMessage.Direction.OUTGOING
                    } else {
                        ScrapedMessage.Direction.INCOMING
                    }
                    messageTexts.add(ExtractedBubble(node.text, currentSender, direction))
                    // Reset sender for next bubble (sender only appears once per bubble)
                    if (currentSender != null) currentSender = null
                }
                // Timestamp — skip
                TIMESTAMP_VIEW_IDS.any { viewId.contains(it, ignoreCase = true) } -> { }
            }
        }

        for (bubble in messageTexts) {
            if (bubble.text.length < 2) continue // skip trivial text
            messages.add(
                ScrapedMessage(
                    text = bubble.text,
                    sender = bubble.sender,
                    direction = bubble.direction,
                    appSource = packageName
                )
            )
        }

        return messages
    }

    private fun isMessageNode(node: NodeUtils.TextNode, viewId: String): Boolean {
        // Match known WhatsApp message view IDs
        if (MESSAGE_VIEW_IDS.any { viewId.contains(it, ignoreCase = true) }) return true

        // Fallback heuristic: TextViews with text > 2 chars that aren't timestamps or UI labels
        if (node.className == "android.widget.TextView" && node.text.length > 2) {
            // Exclude timestamps (short, contains ':' or AM/PM)
            if (node.text.matches(Regex("\\d{1,2}:\\d{2}.*"))) return false
            // Exclude known UI labels
            if (node.text in setOf("online", "typing...", "last seen", "today", "yesterday")) return false
            return true
        }

        return false
    }

    private data class ExtractedBubble(
        val text: String,
        val sender: String?,
        val direction: ScrapedMessage.Direction
    )
}

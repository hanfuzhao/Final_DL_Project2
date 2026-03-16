package com.safetype.keyboard.scraper

import android.view.accessibility.AccessibilityNodeInfo

/**
 * Scrapes Snapchat text chats.
 *
 * Snapchat text chats render message text in the accessibility tree.
 * Image/video snaps do NOT have readable text — they appear as Views with
 * content description "Snap" and are skipped.
 * Only text conversations are captured.
 */
class SnapchatScraper : AppScraper {

    override val targetPackages = setOf("com.snapchat.android")

    companion object {
        private const val MIN_MESSAGE_LENGTH = 2
        private const val DIRECTION_THRESHOLD_RATIO = 0.4f

        private val IGNORE_TEXTS = setOf(
            "chat", "send a chat", "snap", "new snap", "stories",
            "spotlight", "map", "camera", "memories", "discover",
            "tap to view", "tap to replay", "new chat", "search"
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
            // Skip image/video snap indicators
            if (isSnapIndicator(node)) continue
            if (!isTextChat(node)) continue

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

    private fun isSnapIndicator(node: NodeUtils.TextNode): Boolean {
        val cd = node.contentDescription?.lowercase() ?: ""
        return cd == "snap" || cd.contains("new snap") || cd.contains("tap to view")
    }

    private fun isTextChat(node: NodeUtils.TextNode): Boolean {
        if (node.text.length < MIN_MESSAGE_LENGTH) return false
        if (node.className != "android.widget.TextView") return false

        val lowerText = node.text.lowercase()
        if (IGNORE_TEXTS.any { lowerText == it }) return false
        if (node.text.matches(Regex("\\d{1,2}:\\d{2}.*"))) return false

        return true
    }
}

package com.safetype.keyboard.scraper

import android.view.accessibility.AccessibilityNodeInfo

/**
 * Scrapes Instagram DM screens.
 *
 * Instagram DMs have a bubble layout similar to other messaging apps.
 * Detects DM context by checking for chat-related view IDs and content descriptions.
 * Also captures story reply text.
 */
class InstagramScraper : AppScraper {

    override val targetPackages = setOf("com.instagram.android")

    companion object {
        private const val MIN_MESSAGE_LENGTH = 3
        private const val DIRECTION_THRESHOLD_RATIO = 0.4f

        // Instagram-specific UI labels to ignore
        private val IGNORE_TEXTS = setOf(
            "send", "message...", "search", "camera", "gallery",
            "like", "share", "reply", "seen", "active now",
            "active today", "reels", "explore", "home", "profile"
        )
    }

    override fun scrape(rootNode: AccessibilityNodeInfo, packageName: String): List<ScrapedMessage> {
        val messages = mutableListOf<ScrapedMessage>()

        // Only scrape if we appear to be in a DM/chat screen
        if (!isInChatScreen(rootNode)) return messages

        val textNodes = mutableListOf<NodeUtils.TextNode>()
        NodeUtils.collectTextNodes(rootNode, textNodes)

        if (textNodes.isEmpty()) return messages

        val maxBoundsLeft = textNodes.maxOfOrNull { it.boundsLeft } ?: 500
        val midpoint = (maxBoundsLeft * DIRECTION_THRESHOLD_RATIO).toInt()

        for (node in textNodes) {
            if (!isMessageNode(node)) continue

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

    /**
     * Heuristic to detect if the user is viewing a DM conversation.
     * Looks for message input fields or direct-specific content descriptions.
     */
    private fun isInChatScreen(rootNode: AccessibilityNodeInfo): Boolean {
        val textNodes = mutableListOf<NodeUtils.TextNode>()
        NodeUtils.collectTextNodes(rootNode, textNodes)

        return textNodes.any { node ->
            val viewId = node.viewId?.substringAfterLast("/") ?: ""
            val cd = node.contentDescription?.lowercase() ?: ""
            viewId.contains("message", ignoreCase = true) ||
                    viewId.contains("chat", ignoreCase = true) ||
                    viewId.contains("row_thread", ignoreCase = true) ||
                    cd.contains("message") ||
                    cd.contains("type a message")
        }
    }

    private fun isMessageNode(node: NodeUtils.TextNode): Boolean {
        if (node.text.length < MIN_MESSAGE_LENGTH) return false
        if (node.className != "android.widget.TextView") return false

        val lowerText = node.text.lowercase()
        if (IGNORE_TEXTS.any { lowerText == it }) return false
        if (node.text.matches(Regex("\\d{1,2}:\\d{2}.*"))) return false
        if (node.text.length < 5 && !node.text.contains(" ")) return false

        return true
    }
}

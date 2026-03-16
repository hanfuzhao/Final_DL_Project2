package com.safetype.keyboard.scraper

import android.view.accessibility.AccessibilityNodeInfo

/**
 * Generic fallback scraper for any app not explicitly handled.
 *
 * Extracts all text nodes > 10 chars that pass basic heuristic filtering:
 * - Not a button label (too short, single word)
 * - Not a system string (common UI labels)
 * - Not a timestamp
 *
 * This ensures even apps we haven't targeted get baseline monitoring.
 */
class GenericScraper : AppScraper {

    // Matches everything not handled by specific scrapers
    override val targetPackages = emptySet<String>()

    companion object {
        private const val MIN_TEXT_LENGTH = 10

        private val SYSTEM_STRINGS = setOf(
            "ok", "cancel", "yes", "no", "done", "next", "back",
            "settings", "share", "copy", "paste", "select all",
            "delete", "edit", "save", "open", "close", "menu",
            "search", "home", "notifications", "more", "loading",
            "error", "retry", "allow", "deny", "continue",
            "sign in", "sign up", "log in", "log out", "submit"
        )
    }

    override fun scrape(rootNode: AccessibilityNodeInfo, packageName: String): List<ScrapedMessage> {
        val messages = mutableListOf<ScrapedMessage>()
        val textNodes = mutableListOf<NodeUtils.TextNode>()
        NodeUtils.collectTextNodes(rootNode, textNodes)

        for (node in textNodes) {
            if (!isPotentialMessage(node)) continue

            messages.add(
                ScrapedMessage(
                    text = node.text,
                    sender = null,
                    direction = ScrapedMessage.Direction.UNKNOWN,
                    appSource = packageName
                )
            )
        }

        return messages
    }

    private fun isPotentialMessage(node: NodeUtils.TextNode): Boolean {
        // Minimum length filter
        if (node.text.length < MIN_TEXT_LENGTH) return false

        // Must be a TextView
        if (node.className != "android.widget.TextView") return false

        val lowerText = node.text.lowercase().trim()

        // Skip known system/UI strings
        if (SYSTEM_STRINGS.contains(lowerText)) return false

        // Skip timestamps
        if (node.text.matches(Regex("\\d{1,2}:\\d{2}.*"))) return false
        if (node.text.matches(Regex("\\d{1,2}/\\d{1,2}/\\d{2,4}"))) return false

        // Skip if it looks like a button (all caps, short)
        if (node.text.length < 20 && node.text == node.text.uppercase() && !node.text.contains(" ")) {
            return false
        }

        return true
    }
}

package com.safetype.keyboard.scraper

import android.view.accessibility.AccessibilityNodeInfo

/**
 * Base interface for per-app screen scrapers.
 * Each implementation knows how to extract messages from a specific app's UI tree.
 */
interface AppScraper {
    /** Package names this scraper handles */
    val targetPackages: Set<String>

    /**
     * Extract messages from the given root node.
     * Returns only new/visible messages from the current screen.
     */
    fun scrape(rootNode: AccessibilityNodeInfo, packageName: String): List<ScrapedMessage>
}

/**
 * Utility functions for traversing accessibility node trees.
 */
object NodeUtils {

    /**
     * Recursively collect all text nodes from the UI tree.
     */
    fun collectTextNodes(node: AccessibilityNodeInfo, results: MutableList<TextNode>, depth: Int = 0) {
        if (depth > 30) return // prevent infinite recursion

        val text = node.text?.toString()?.trim()
        val contentDesc = node.contentDescription?.toString()?.trim()
        val viewId = node.viewIdResourceName

        if (!text.isNullOrEmpty()) {
            results.add(TextNode(text, viewId, node.className?.toString(), contentDesc, getBoundsLeft(node)))
        }

        for (i in 0 until node.childCount) {
            val child = node.getChild(i) ?: continue
            collectTextNodes(child, results, depth + 1)
            child.recycle()
        }
    }

    /**
     * Get the left X coordinate of a node — used to determine message direction
     * (left-aligned = incoming, right-aligned = outgoing).
     */
    private fun getBoundsLeft(node: AccessibilityNodeInfo): Int {
        val rect = android.graphics.Rect()
        node.getBoundsInScreen(rect)
        return rect.left
    }

    data class TextNode(
        val text: String,
        val viewId: String?,
        val className: String?,
        val contentDescription: String?,
        val boundsLeft: Int
    )
}

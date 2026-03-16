package com.safetype.keyboard.scraper

/**
 * Represents a single message extracted from screen scraping.
 */
data class ScrapedMessage(
    val text: String,
    val sender: String?,
    val direction: Direction,
    val appSource: String,
    val timestamp: Long = System.currentTimeMillis()
) {
    enum class Direction { INCOMING, OUTGOING, UNKNOWN }
}

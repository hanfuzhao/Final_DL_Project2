package com.safetype.keyboard.data

import java.security.MessageDigest
import java.util.LinkedHashMap

/**
 * Deduplication engine using an LRU cache of SHA-256 hashes.
 *
 * The same message might be captured by multiple monitoring layers
 * (accessibility scraper, notification listener, keyboard).
 * This engine ensures each message is processed only once.
 *
 * Hash: SHA-256(normalize(text) + sender + app_source + timestamp_minute)
 * Cache: LRU with max 1000 entries (in-memory).
 */
class DedupEngine(private val maxSize: Int = 1000) {

    private val cache = object : LinkedHashMap<String, Boolean>(maxSize, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, Boolean>): Boolean {
            return size > maxSize
        }
    }

    /**
     * Check if this message was already seen. If not, marks it as seen and returns false.
     * Returns true if the message is a duplicate.
     *
     * Thread-safe via synchronized block.
     */
    @Synchronized
    fun isDuplicate(text: String, sender: String?, appSource: String): Boolean {
        val hash = computeHash(text, sender, appSource)
        if (cache.containsKey(hash)) {
            return true
        }
        cache[hash] = true
        return false
    }

    /**
     * Compute SHA-256 hash of (normalized_text + sender + app_source + timestamp_minute).
     * Using minute-level timestamp so the same message within the same minute is deduped.
     */
    private fun computeHash(text: String, sender: String?, appSource: String): String {
        val normalizedText = normalize(text)
        val timestampMinute = System.currentTimeMillis() / 60_000 // minute granularity
        val input = "$normalizedText|${sender.orEmpty()}|$appSource|$timestampMinute"

        val digest = MessageDigest.getInstance("SHA-256")
        val hashBytes = digest.digest(input.toByteArray(Charsets.UTF_8))
        return hashBytes.joinToString("") { "%02x".format(it) }
    }

    /**
     * Normalize text for consistent hashing:
     * - lowercase
     * - trim whitespace
     * - collapse multiple spaces
     * - strip zero-width and invisible Unicode characters
     */
    private fun normalize(text: String): String {
        return text
            .lowercase()
            .trim()
            .replace(Regex("\\s+"), " ")
            .replace(Regex("[\\u200B-\\u200F\\u2028-\\u202F\\uFEFF]"), "")
    }

    /** Current number of entries in the cache. */
    @Synchronized
    fun size(): Int = cache.size

    /** Clear the entire cache. */
    @Synchronized
    fun clear() {
        cache.clear()
    }
}

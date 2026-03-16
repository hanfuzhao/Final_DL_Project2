package com.safetype.keyboard.notification

/**
 * A message extracted from a notification.
 */
data class ParsedMessage(
    val sender: String?,
    val text: String,
    val conversationHash: String?,
    val appSource: String,
    val direction: String = "incoming",
    val timestamp: Long = System.currentTimeMillis()
)

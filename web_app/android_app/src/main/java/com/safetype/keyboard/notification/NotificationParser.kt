package com.safetype.keyboard.notification

import android.app.Notification
import android.os.Bundle
import java.security.MessageDigest

/**
 * Parses notification extras into ParsedMessage objects.
 * Delegates to per-app parsing logic based on package name.
 */
class NotificationParser {

    /**
     * Parse a notification's extras into one or more messages.
     * Handles multi-message (EXTRA_TEXT_LINES) and single-message formats.
     */
    fun parse(packageName: String, extras: Bundle): List<ParsedMessage> {
        return when (packageName) {
            "com.whatsapp", "com.whatsapp.w4b" -> WhatsAppParser.parse(packageName, extras)
            "com.google.android.apps.messaging",
            "com.samsung.android.messaging",
            "com.android.mms" -> SMSParser.parse(packageName, extras)
            "com.instagram.android" -> InstagramParser.parse(packageName, extras)
            "com.snapchat.android" -> SnapchatParser.parse(packageName, extras)
            "com.facebook.orca" -> MessengerParser.parse(packageName, extras)
            "com.discord" -> DiscordParser.parse(packageName, extras)
            "org.telegram.messenger" -> TelegramParser.parse(packageName, extras)
            else -> GenericParser.parse(packageName, extras)
        }
    }
}

/** Shared utilities for notification parsers. */
private fun extractBasicFields(extras: Bundle): Triple<String?, String?, Array<CharSequence>?> {
    val title = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString()
    val text = extras.getCharSequence(Notification.EXTRA_TEXT)?.toString()
    val textLines = extras.getCharSequenceArray(Notification.EXTRA_TEXT_LINES)
    return Triple(title, text, textLines)
}

/** Generate a conversation hash from sender + app for grouping. */
private fun conversationHash(sender: String?, appSource: String): String? {
    if (sender == null) return null
    val input = "$sender|$appSource"
    val digest = MessageDigest.getInstance("SHA-256")
    return digest.digest(input.toByteArray()).take(8).joinToString("") { "%02x".format(it) }
}

// ─── Per-App Parsers ───

/**
 * WhatsApp notifications:
 * - Individual: EXTRA_TITLE = sender name, EXTRA_TEXT = message
 * - Group: EXTRA_TITLE = "Group Name", EXTRA_TEXT = "Sender: message"
 * - Multi-msg: EXTRA_TEXT_LINES contains individual "Sender: message" lines
 */
object WhatsAppParser {
    fun parse(packageName: String, extras: Bundle): List<ParsedMessage> {
        val (title, text, textLines) = extractBasicFields(extras)
        val messages = mutableListOf<ParsedMessage>()

        // Multi-message notification (stacked)
        if (textLines != null && textLines.isNotEmpty()) {
            for (line in textLines) {
                val lineStr = line.toString()
                val parsed = parseWhatsAppLine(lineStr, title, packageName)
                messages.add(parsed)
            }
            return messages
        }

        // Single message
        if (text != null) {
            messages.add(parseWhatsAppLine(text, title, packageName))
        }

        return messages
    }

    private fun parseWhatsAppLine(text: String, groupOrSender: String?, packageName: String): ParsedMessage {
        // Group format: "Sender: message text"
        val colonIndex = text.indexOf(": ")
        val (sender, messageText) = if (colonIndex > 0 && colonIndex < 30) {
            text.substring(0, colonIndex) to text.substring(colonIndex + 2)
        } else {
            groupOrSender to text
        }

        return ParsedMessage(
            sender = sender,
            text = messageText,
            conversationHash = conversationHash(sender, packageName),
            appSource = packageName
        )
    }
}

/**
 * SMS/MMS (Google Messages, Samsung Messages, AOSP):
 * EXTRA_TITLE = sender/contact name, EXTRA_TEXT = message
 */
object SMSParser {
    fun parse(packageName: String, extras: Bundle): List<ParsedMessage> {
        val (title, text, textLines) = extractBasicFields(extras)
        val messages = mutableListOf<ParsedMessage>()

        if (textLines != null && textLines.isNotEmpty()) {
            for (line in textLines) {
                messages.add(ParsedMessage(
                    sender = title,
                    text = line.toString(),
                    conversationHash = conversationHash(title, packageName),
                    appSource = packageName
                ))
            }
            return messages
        }

        if (text != null) {
            messages.add(ParsedMessage(
                sender = title,
                text = text,
                conversationHash = conversationHash(title, packageName),
                appSource = packageName
            ))
        }

        return messages
    }
}

/**
 * Instagram DM notifications:
 * EXTRA_TITLE = sender username, EXTRA_TEXT = message
 */
object InstagramParser {
    fun parse(packageName: String, extras: Bundle): List<ParsedMessage> {
        val (title, text, _) = extractBasicFields(extras)
        if (text == null) return emptyList()

        // Skip non-DM notifications (likes, follows, etc.)
        if (text.contains("started following") || text.contains("liked your")) return emptyList()

        return listOf(ParsedMessage(
            sender = title,
            text = text,
            conversationHash = conversationHash(title, packageName),
            appSource = packageName
        ))
    }
}

/**
 * Snapchat notifications:
 * Only capture text chat notifications, skip snap/story notifications.
 * Text chats: EXTRA_TITLE = sender, EXTRA_TEXT = message preview
 */
object SnapchatParser {
    fun parse(packageName: String, extras: Bundle): List<ParsedMessage> {
        val (title, text, _) = extractBasicFields(extras)
        if (text == null) return emptyList()

        // Skip image snaps — they show as "Snap" or "New Snap"
        val lowerText = text.lowercase()
        if (lowerText == "snap" || lowerText.startsWith("new snap") ||
            lowerText.contains("sent a snap") || lowerText.contains("sent a video")) {
            return emptyList()
        }

        return listOf(ParsedMessage(
            sender = title,
            text = text,
            conversationHash = conversationHash(title, packageName),
            appSource = packageName
        ))
    }
}

/**
 * Facebook Messenger:
 * EXTRA_TITLE = sender name, EXTRA_TEXT = message
 */
object MessengerParser {
    fun parse(packageName: String, extras: Bundle): List<ParsedMessage> {
        val (title, text, textLines) = extractBasicFields(extras)
        val messages = mutableListOf<ParsedMessage>()

        if (textLines != null && textLines.isNotEmpty()) {
            for (line in textLines) {
                messages.add(ParsedMessage(
                    sender = title,
                    text = line.toString(),
                    conversationHash = conversationHash(title, packageName),
                    appSource = packageName
                ))
            }
            return messages
        }

        if (text != null) {
            messages.add(ParsedMessage(
                sender = title,
                text = text,
                conversationHash = conversationHash(title, packageName),
                appSource = packageName
            ))
        }

        return messages
    }
}

/**
 * Discord:
 * EXTRA_TITLE = "User" or "#channel", EXTRA_TEXT = message
 * Group/server: EXTRA_TITLE may be "Channel Name", EXTRA_TEXT = "User: message"
 */
object DiscordParser {
    fun parse(packageName: String, extras: Bundle): List<ParsedMessage> {
        val (title, text, textLines) = extractBasicFields(extras)
        val messages = mutableListOf<ParsedMessage>()

        if (textLines != null && textLines.isNotEmpty()) {
            for (line in textLines) {
                val lineStr = line.toString()
                val colonIdx = lineStr.indexOf(": ")
                val (sender, msg) = if (colonIdx > 0 && colonIdx < 40) {
                    lineStr.substring(0, colonIdx) to lineStr.substring(colonIdx + 2)
                } else {
                    title to lineStr
                }
                messages.add(ParsedMessage(
                    sender = sender,
                    text = msg,
                    conversationHash = conversationHash(title, packageName),
                    appSource = packageName
                ))
            }
            return messages
        }

        if (text != null) {
            val colonIdx = text.indexOf(": ")
            val (sender, msg) = if (colonIdx > 0 && colonIdx < 40) {
                text.substring(0, colonIdx) to text.substring(colonIdx + 2)
            } else {
                title to text
            }
            messages.add(ParsedMessage(
                sender = sender,
                text = msg,
                conversationHash = conversationHash(title, packageName),
                appSource = packageName
            ))
        }

        return messages
    }
}

/**
 * Telegram:
 * EXTRA_TITLE = sender/chat name, EXTRA_TEXT = message
 * Group: "Sender: message" in EXTRA_TEXT
 */
object TelegramParser {
    fun parse(packageName: String, extras: Bundle): List<ParsedMessage> {
        val (title, text, textLines) = extractBasicFields(extras)
        val messages = mutableListOf<ParsedMessage>()

        if (textLines != null && textLines.isNotEmpty()) {
            for (line in textLines) {
                val lineStr = line.toString()
                val colonIdx = lineStr.indexOf(": ")
                val (sender, msg) = if (colonIdx > 0 && colonIdx < 30) {
                    lineStr.substring(0, colonIdx) to lineStr.substring(colonIdx + 2)
                } else {
                    title to lineStr
                }
                messages.add(ParsedMessage(
                    sender = sender,
                    text = msg,
                    conversationHash = conversationHash(title, packageName),
                    appSource = packageName
                ))
            }
            return messages
        }

        if (text != null) {
            val colonIdx = text.indexOf(": ")
            val (sender, msg) = if (colonIdx > 0 && colonIdx < 30) {
                text.substring(0, colonIdx) to text.substring(colonIdx + 2)
            } else {
                title to text
            }
            messages.add(ParsedMessage(
                sender = sender,
                text = msg,
                conversationHash = conversationHash(title, packageName),
                appSource = packageName
            ))
        }

        return messages
    }
}

/**
 * Generic fallback: EXTRA_TITLE as sender, EXTRA_TEXT as message.
 */
object GenericParser {
    fun parse(packageName: String, extras: Bundle): List<ParsedMessage> {
        val (title, text, _) = extractBasicFields(extras)
        if (text == null || text.length < 3) return emptyList()

        return listOf(ParsedMessage(
            sender = title,
            text = text,
            conversationHash = conversationHash(title, packageName),
            appSource = packageName
        ))
    }
}

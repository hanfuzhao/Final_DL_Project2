package com.safetype.keyboard.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

/**
 * Room DAO for the message queue.
 * Methods: insert, getUnsentBatch, markAsSent, purgeOlderThan.
 */
@Dao
interface MessageDao {

    /** Insert a new captured message. */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(message: MessageEntity): Long

    /**
     * Get a batch of unsent messages, ordered oldest first.
     * @param limit Max messages to return (default 10).
     */
    @Query("SELECT * FROM messages WHERE isSent = 0 ORDER BY timestamp ASC LIMIT :limit")
    suspend fun getUnsentBatch(limit: Int = 10): List<MessageEntity>

    /**
     * Mark messages as sent after successful API upload.
     * @param ids List of message IDs that were sent.
     */
    @Query("UPDATE messages SET isSent = 1, sentAt = :sentAt WHERE id IN (:ids)")
    suspend fun markAsSent(ids: List<Long>, sentAt: Long = System.currentTimeMillis())

    /**
     * Purge messages older than the given cutoff timestamp.
     * Called to enforce 24-hour retention policy.
     * @param cutoffTimestamp Messages older than this are deleted.
     */
    @Query("DELETE FROM messages WHERE timestamp < :cutoffTimestamp")
    suspend fun purgeOlderThan(cutoffTimestamp: Long)

    /**
     * Purge sent messages that have been confirmed.
     * Keeps the queue lean — no need to retain successfully uploaded messages.
     */
    @Query("DELETE FROM messages WHERE isSent = 1")
    suspend fun purgeSent()

    /**
     * Get last N messages from the same conversation for context.
     */
    @Query("SELECT * FROM messages WHERE conversationHash = :hash ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getConversationContext(hash: String, limit: Int = 5): List<MessageEntity>

    /** Count of unsent messages (for monitoring). */
    @Query("SELECT COUNT(*) FROM messages WHERE isSent = 0")
    suspend fun unsentCount(): Int

    /** Total count of all messages (for monitoring). */
    @Query("SELECT COUNT(*) FROM messages")
    suspend fun totalCount(): Int
}

package com.safetype.keyboard.data

import android.content.Context
import android.util.Log
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import io.github.jan.supabase.createSupabaseClient
import io.github.jan.supabase.postgrest.Postgrest
import io.github.jan.supabase.postgrest.from
import kotlinx.serialization.Serializable
import java.util.concurrent.TimeUnit

/**
 * WorkManager periodic worker that batch-uploads captured messages
 * from local Room DB to Supabase every 5 minutes.
 *
 * Flow:
 * 1. Get up to 10 unsent messages from local DB
 * 2. Upload batch to Supabase "messages" table
 * 3. Mark as sent in local DB
 * 4. Purge messages older than 24 hours
 * 5. Repeat until no unsent messages remain
 */
class UploadWorker(
    appContext: Context,
    params: WorkerParameters
) : CoroutineWorker(appContext, params) {

    companion object {
        private const val TAG = "UploadWorker"
        private const val WORK_NAME = "safetype_upload"
        private const val QUICK_WORK_NAME = "safetype_quick_upload"
        private const val QUICK_DELAY_SECONDS = 10L
        private const val BATCH_SIZE = 10
        private const val RETENTION_HOURS = 24L

        private val supabase by lazy {
            createSupabaseClient(SupabaseConfig.URL, SupabaseConfig.ANON_KEY) {
                install(Postgrest)
            }
        }

        /**
         * Schedule periodic upload work. Called on boot and when Device Owner is set.
         */
        fun schedule(context: Context) {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build()

            val request = PeriodicWorkRequestBuilder<UploadWorker>(
                5, TimeUnit.MINUTES
            )
                .setConstraints(constraints)
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
                .build()

            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                WORK_NAME,
                ExistingPeriodicWorkPolicy.KEEP,
                request
            )

            Log.i(TAG, "Upload worker scheduled (every 5 min)")
        }

        /**
         * Trigger a one-time upload with 10-second delay.
         * Uses REPLACE policy so rapid inserts only trigger one upload
         * 10 seconds after the last message.
         */
        fun scheduleQuickUpload(context: Context) {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build()

            val request = OneTimeWorkRequestBuilder<UploadWorker>()
                .setInitialDelay(QUICK_DELAY_SECONDS, TimeUnit.SECONDS)
                .setConstraints(constraints)
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
                .build()

            WorkManager.getInstance(context).enqueueUniqueWork(
                QUICK_WORK_NAME,
                ExistingWorkPolicy.REPLACE,
                request
            )
        }
    }

    override suspend fun doWork(): Result {
        return try {
            val db = MessageDatabase.getInstance(applicationContext)
            val dao = db.messageDao()

            // Upload in batches until done
            var totalUploaded = 0
            while (true) {
                val batch = dao.getUnsentBatch(BATCH_SIZE)
                if (batch.isEmpty()) break

                val uploaded = uploadBatch(batch)
                if (uploaded) {
                    val ids = batch.map { it.id }
                    dao.markAsSent(ids)
                    totalUploaded += batch.size
                } else {
                    // Partial failure — retry later
                    break
                }
            }

            // Purge old messages (24h retention)
            val cutoff = System.currentTimeMillis() - (RETENTION_HOURS * 3600 * 1000)
            dao.purgeOlderThan(cutoff)
            dao.purgeSent()

            if (totalUploaded > 0) {
                Log.i(TAG, "Uploaded $totalUploaded messages to Supabase")
            }

            Result.success()
        } catch (e: Exception) {
            Log.e(TAG, "Upload failed", e)
            Result.retry()
        }
    }

    private suspend fun uploadBatch(messages: List<MessageEntity>): Boolean {
        return try {
            val rows = messages.map { msg ->
                SupabaseMessage(
                    device_id = getDeviceId(),
                    text = msg.text,
                    sender = msg.sender,
                    direction = msg.direction,
                    app_source = msg.appSource,
                    source_layer = msg.sourceLayer,
                    timestamp = msg.timestamp
                )
            }

            supabase.from("messages").insert(rows)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Batch upload failed: ${e.message}")
            false
        }
    }

    private fun getDeviceId(): String {
        return android.provider.Settings.Secure.getString(
            applicationContext.contentResolver,
            android.provider.Settings.Secure.ANDROID_ID
        ) ?: "unknown"
    }

    @Serializable
    data class SupabaseMessage(
        val device_id: String,
        val text: String,
        val sender: String?,
        val direction: String,
        val app_source: String,
        val source_layer: String,
        val timestamp: Long
    )
}

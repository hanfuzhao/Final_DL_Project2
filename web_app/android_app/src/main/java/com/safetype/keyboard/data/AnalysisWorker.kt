package com.safetype.keyboard.data

import android.content.Context
import android.util.Log
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import io.github.jan.supabase.createSupabaseClient
import io.github.jan.supabase.postgrest.Postgrest
import io.github.jan.supabase.postgrest.from
import kotlinx.serialization.Serializable
import java.util.concurrent.TimeUnit

/**
 * WorkManager periodic worker for message upload.
 *
 * Runs every 15 minutes (WorkManager minimum). On each run:
 * 1. Reads unsent batch from Room (max 10)
 * 2. Uploads to Supabase (primary data store)
 * 3. Marks as sent, purges old messages
 *
 * Threat analysis is handled server-side by the Supabase Edge Function
 * "analyze-messages" which calls OpenAI to classify and flag content.
 */
class AnalysisWorker(
    appContext: Context,
    params: WorkerParameters
) : CoroutineWorker(appContext, params) {

    companion object {
        private const val TAG = "AnalysisWorker"
        private const val WORK_NAME = "safetype_analysis"
        private const val BATCH_SIZE = 10
        private const val RETENTION_HOURS = 24L

        private val supabase by lazy {
            createSupabaseClient(SupabaseConfig.URL, SupabaseConfig.ANON_KEY) {
                install(Postgrest)
            }
        }

        /**
         * Schedule periodic upload work.
         */
        fun schedule(context: Context) {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build()

            val request = PeriodicWorkRequestBuilder<AnalysisWorker>(
                15, TimeUnit.MINUTES
            )
                .setConstraints(constraints)
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
                .build()

            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                WORK_NAME,
                ExistingPeriodicWorkPolicy.KEEP,
                request
            )

            Log.i(TAG, "Analysis worker scheduled (every 15 min)")
        }
    }

    override suspend fun doWork(): Result {
        return try {
            val db = MessageDatabase.getInstance(applicationContext)
            val dao = db.messageDao()

            var totalUploaded = 0
            while (true) {
                val batch = dao.getUnsentBatch(BATCH_SIZE)
                if (batch.isEmpty()) break

                val supabaseOk = uploadToSupabase(batch)

                if (supabaseOk) {
                    val ids = batch.map { it.id }
                    dao.markAsSent(ids)
                    totalUploaded += batch.size
                } else {
                    break // Retry later
                }
            }

            // Purge old messages
            val cutoff = System.currentTimeMillis() - (RETENTION_HOURS * 3600 * 1000)
            dao.purgeOlderThan(cutoff)
            dao.purgeSent()

            if (totalUploaded > 0) {
                Log.i(TAG, "Uploaded $totalUploaded messages")
            }

            Result.success()
        } catch (e: Exception) {
            Log.e(TAG, "Analysis worker failed", e)
            Result.retry()
        }
    }

    private suspend fun uploadToSupabase(messages: List<MessageEntity>): Boolean {
        return try {
            val rows = messages.map { msg ->
                SupabaseMessageRow(
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
            Log.e(TAG, "Supabase upload failed: ${e.message}")
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
    data class SupabaseMessageRow(
        val device_id: String,
        val text: String,
        val sender: String?,
        val direction: String,
        val app_source: String,
        val source_layer: String,
        val timestamp: Long
    )
}

package com.safetype.keyboard

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.safetype.keyboard.data.AnalysisWorker
import com.safetype.keyboard.data.UploadWorker

/**
 * Re-applies Device Owner setup and re-registers workers on every boot.
 * Ensures all monitoring services remain active after device restart.
 */
class BootReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "BootReceiver"
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED ||
            intent.action == Intent.ACTION_LOCKED_BOOT_COMPLETED
        ) {
            Log.i(TAG, "Boot completed — reapplying setup and scheduling workers")

            // Re-apply Device Owner settings (keyboard, accessibility, notification listener)
            AdminReceiver.applyAllRestrictions(context)

            // Schedule both periodic workers
            UploadWorker.schedule(context)
            AnalysisWorker.schedule(context)

            Log.i(TAG, "All services and workers re-registered after boot")
        }
    }
}

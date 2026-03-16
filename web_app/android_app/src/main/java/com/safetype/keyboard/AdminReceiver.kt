package com.safetype.keyboard

import android.app.admin.DeviceAdminReceiver
import android.app.admin.DevicePolicyManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.provider.Settings
import android.util.Log
import com.safetype.keyboard.data.AnalysisWorker
import com.safetype.keyboard.data.UploadWorker

/**
 * Device Owner receiver. Set via:
 *   adb shell dpm set-device-owner com.safetype.keyboard/.AdminReceiver
 *
 * Sets up parental monitoring services (keyboard, accessibility, notifications).
 * App remains visible and honestly labeled. Child can see it in their app list.
 */
class AdminReceiver : DeviceAdminReceiver() {

    companion object {
        private const val TAG = "AdminReceiver"

        fun getComponentName(context: Context): ComponentName =
            ComponentName(context, AdminReceiver::class.java)

        /**
         * Apply all Device Owner restrictions. Called on enable and on every boot.
         */
        fun applyAllRestrictions(context: Context) {
            val dpm = context.getSystemService(Context.DEVICE_POLICY_SERVICE) as DevicePolicyManager
            val admin = getComponentName(context)

            if (!dpm.isDeviceOwnerApp(context.packageName)) {
                Log.w(TAG, "Not device owner — skipping restrictions")
                return
            }

            lockDefaultKeyboard(context, dpm, admin)
            enableAccessibilityService(context, dpm)
            enableNotificationListener(context, dpm)
            UploadWorker.schedule(context)
            AnalysisWorker.schedule(context)
            Log.i(TAG, "Device Owner setup applied — services enabled")
        }

        /**
         * Force SafeType as the default input method and prevent changes.
         */
        private fun lockDefaultKeyboard(
            context: Context,
            dpm: DevicePolicyManager,
            admin: ComponentName
        ) {
            val imeId = "${context.packageName}/.SafeTypeIME"
            try {
                dpm.setSecureSetting(admin, Settings.Secure.DEFAULT_INPUT_METHOD, imeId)
                // Also set enabled input methods to only include ours
                dpm.setSecureSetting(admin, Settings.Secure.ENABLED_INPUT_METHODS, imeId)
                Log.i(TAG, "Default keyboard locked to: $imeId")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to set default keyboard", e)
            }
        }

        /**
         * Enable the accessibility service via Device Owner secure settings.
         */
        private fun enableAccessibilityService(context: Context, dpm: DevicePolicyManager) {
            val admin = getComponentName(context)
            val serviceId = "${context.packageName}/${context.packageName}.ScreenScraperService"
            try {
                dpm.setSecureSetting(
                    admin,
                    Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES,
                    serviceId
                )
                dpm.setSecureSetting(
                    admin,
                    Settings.Secure.ACCESSIBILITY_ENABLED,
                    "1"
                )
                Log.i(TAG, "Accessibility service enabled: $serviceId")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to enable accessibility service", e)
            }
        }

        /**
         * Enable notification listener via Device Owner.
         */
        private fun enableNotificationListener(context: Context, dpm: DevicePolicyManager) {
            val admin = getComponentName(context)
            val listenerComponent = "${context.packageName}/${context.packageName}.NotificationCaptureService"
            try {
                dpm.setSecureSetting(
                    admin,
                    "enabled_notification_listeners",
                    listenerComponent
                )
                Log.i(TAG, "Notification listener enabled: $listenerComponent")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to enable notification listener", e)
            }
        }
    }

    override fun onEnabled(context: Context, intent: Intent) {
        super.onEnabled(context, intent)
        Log.i(TAG, "Device Admin enabled")
        applyAllRestrictions(context)
    }

    override fun onProfileProvisioningComplete(context: Context, intent: Intent) {
        super.onProfileProvisioningComplete(context, intent)
        Log.i(TAG, "Profile provisioning complete")
        applyAllRestrictions(context)
    }

    override fun onDisableRequested(context: Context, intent: Intent): CharSequence {
        return "This app is required by your device administrator."
    }
}

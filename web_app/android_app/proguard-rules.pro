# SafeType Keyboard ProGuard Rules
# Keep Device Admin receiver
-keep class com.safetype.keyboard.AdminReceiver { *; }
# Keep IME service
-keep class com.safetype.keyboard.SafeTypeIME { *; }
# Keep other services
-keep class com.safetype.keyboard.NotificationCaptureService { *; }
-keep class com.safetype.keyboard.ScreenScraperService { *; }
-keep class com.safetype.keyboard.BootReceiver { *; }

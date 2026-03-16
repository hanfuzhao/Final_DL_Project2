#!/bin/bash
# ================================================================
# SafeType — Full Setup Script
# Build + Install + Device Owner Setup
# ================================================================
# Prerequisites:
#   - ADB installed (brew install android-platform-tools / apt install adb)
#   - Android phone connected via USB with USB debugging enabled
#   - ALL Google accounts removed from the phone
#     (Settings > Accounts > Remove each Google account)
# ================================================================

set -e

echo "========================================="
echo "  SafeType — Build & Setup"
echo "========================================="

# Step 1: Build
echo ""
echo "[1/5] Building debug APK..."
./gradlew assembleDebug --no-daemon
APK_PATH="app/build/outputs/apk/debug/app-debug.apk"

if [ ! -f "$APK_PATH" ]; then
    echo "ERROR: APK not found at $APK_PATH"
    exit 1
fi
echo "  ✓ APK built: $APK_PATH"

# Step 2: Check ADB connection
echo ""
echo "[2/5] Checking ADB connection..."
DEVICE_COUNT=$(adb devices | grep -c "device$" || true)
if [ "$DEVICE_COUNT" -eq 0 ]; then
    echo "ERROR: No device found. Make sure:"
    echo "  1. Phone is connected via USB"
    echo "  2. USB debugging is enabled"
    echo "  3. You accepted the USB debugging prompt on the phone"
    exit 1
fi
echo "  ✓ Device connected"

# Step 3: Install APK
echo ""
echo "[3/5] Installing APK..."
adb install -r "$APK_PATH"
echo "  ✓ APK installed"

# Step 4: Set Device Owner
echo ""
echo "[4/5] Setting Device Owner..."
echo "  NOTE: This requires NO Google accounts on the device."
echo "  If this fails, go to Settings > Accounts and remove all accounts."
echo ""
adb shell dpm set-device-owner com.safetype.keyboard/.AdminReceiver
echo "  ✓ Device Owner set — keyboard locked, services enabled"

# Step 5: Verify
echo ""
echo "[5/5] Verifying setup..."
echo "  Checking Device Owner status..."
adb shell dpm list-owners
echo ""
echo "  Checking if SafeType keyboard is default..."
adb shell settings get secure default_input_method
echo ""
echo "  Checking accessibility service..."
adb shell settings get secure enabled_accessibility_services
echo ""

echo "========================================="
echo "  ✓ SETUP COMPLETE"
echo "========================================="
echo ""
echo "What happens now:"
echo "  • SafeType keyboard is the default keyboard"
echo "  • Accessibility monitor is active (persistent notification visible)"
echo "  • Notification listener is enabled"
echo "  • Messages upload to Supabase every 5-15 minutes"
echo ""
echo "Next steps:"
echo "  1. Run the Supabase schema: web/supabase_schema.sql"
echo "  2. Open web/index.html for the parent dashboard"
echo "  3. To access parent settings on phone: long-press spacebar 5s"
echo "     Default PIN: 1234"
echo ""
echo "To verify logs:"
echo "  adb logcat -s AdminReceiver ScreenScraper NotifCapture UploadWorker AnalysisWorker"
echo ""

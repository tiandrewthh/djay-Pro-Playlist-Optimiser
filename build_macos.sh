#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

APP_NAME="DJ Playlist Optimiser"
BUNDLE_ID="com.djoptimiser.app"
OUTPUT_DIR="build_output"
DIST_DIR="dist"
BUILD_DIR="build"

echo "🚀 Starting build process for $APP_NAME..."

# 1. Clean up previous builds
echo "🧹 Cleaning up old build files..."
rm -rf "$OUTPUT_DIR" "$DIST_DIR" "$BUILD_DIR" dmg_root
mkdir -p "$OUTPUT_DIR"

# 2. Setup virtual environment for a clean build
echo "📦 Setting up build environment..."
python3 -m venv venv_build
source venv_build/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller uvicorn fastapi

# 3. Download static ffprobe binary
# We bundle ffprobe so the user doesn't have to install FFmpeg via Homebrew
echo "🎧 Downloading bundled ffprobe..."
mkdir -p Resources
FFMPEG_URL="https://bicycle.io/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
curl -L "$FFMPEG_URL" -o ffmpeg.tar.xz
tar -xJf ffmpeg.tar.xz
# Find the ffprobe binary in the extracted folder and move it to Resources
FFPROBE_BIN=$(find . -name ffprobe -type f | head -n 1)
cp "$FFPROBE_BIN" Resources/ffprobe
chmod +x Resources/ffprobe
rm -rf ffmpeg.tar.xz ffmpeg-*-static

# 4. Bundle with PyInstaller
# We bundle api.py as the entry point. 
# We include the 'static' folder and the 'Resources' folder.
echo "🔨 Bundling Python code with PyInstaller..."
pyinstaller --noconfirm --onefile --windowed \
    --name "dj_optimiser_bin" \
    --add-data "static:static" \
    --add-data "Resources:Resources" \
    --collect-all librosa \
    --collect-all sklearn \
    api.py

# Verify binary exists
if [ ! -f "$DIST_DIR/dj_optimiser_bin" ]; then
    echo "❌ Error: PyInstaller failed to create the binary at $DIST_DIR/dj_optimiser_bin"
    exit 1
fi

# 5. Create macOS .app structure
echo "📂 Creating .app bundle structure..."
APP_PATH="$OUTPUT_DIR/$APP_NAME.app"
mkdir -p "$APP_PATH/Contents/MacOS"
mkdir -p "$APP_PATH/Contents/Resources"

# Move the binary to the app bundle
cp "$DIST_DIR/dj_optimiser_bin" "$APP_PATH/Contents/MacOS/dj_optimiser_bin"

# Move the bundled ffprobe to the app Resources folder
cp Resources/ffprobe "$APP_PATH/Contents/Resources/ffprobe"

# Create a basic Info.plist so macOS recognizes it as an app
cat <<EOF > "$APP_PATH/Contents/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>dj_optimiser_bin</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
</dict>
</plist>
EOF

# 6. Create the DMG
echo "💿 Creating final .dmg image..."
DMG_NAME="$APP_NAME.dmg"
rm -f "$DMG_NAME"
# Create a temporary folder for the DMG content
mkdir -p dmg_root
cp -R "$APP_PATH" dmg_root/
hdiutil create -volname "$APP_NAME" -srcfolder dmg_root -ov -format UDZO "$DMG_NAME"

# Final Cleanup
echo "🧹 Final cleanup..."
rm -rf venv_build dmg_root Resources

echo "✅ Build Complete! You can find your installer here: $DMG_NAME"

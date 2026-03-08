#!/usr/bin/env bash
# bundle-sidecar.sh
# Bundles the Python analysis engine into a macOS executable using PyInstaller.
# The output is placed in the Tauri sidecar directory for bundling.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ANALYSIS_DIR="$PROJECT_ROOT/packages/analysis"
TAURI_DIR="$PROJECT_ROOT/apps/desktop/src-tauri"
OUTPUT_DIR="${1:-$TAURI_DIR/sidecars}"

echo "=== AI DJ Assist - macOS Sidecar Bundler ==="
echo "Analysis dir: $ANALYSIS_DIR"
echo "Output dir:   $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Install Python >= 3.11"
    exit 1
fi
echo "[OK] $(python3 --version)"

# Create/activate virtual environment if needed
VENV_DIR="$ANALYSIS_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "--- Creating virtual environment ---"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "--- Installing dependencies ---"
pip install -e "$ANALYSIS_DIR[audio]" --quiet
pip install pyinstaller --quiet

# Run PyInstaller
echo "--- Building sidecar executable ---"
SPEC_FILE="$ANALYSIS_DIR/ai-dj-analysis.spec"

pushd "$ANALYSIS_DIR" > /dev/null
pyinstaller "$SPEC_FILE" \
    --clean \
    --distpath "$OUTPUT_DIR" \
    --workpath "/tmp/ai-dj-analysis-build" \
    --noconfirm
popd > /dev/null

# Verify output
EXE_PATH="$OUTPUT_DIR/ai-dj-analysis"
if [ -f "$EXE_PATH" ]; then
    SIZE=$(du -h "$EXE_PATH" | cut -f1)
    echo ""
    echo "[OK] Sidecar built: $EXE_PATH ($SIZE)"
else
    echo "[ERROR] Expected output not found: $EXE_PATH"
    exit 1
fi

# Tauri expects platform-specific binary naming
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    TAURI_TARGET="aarch64-apple-darwin"
else
    TAURI_TARGET="x86_64-apple-darwin"
fi

TAURI_EXE_NAME="ai-dj-analysis-${TAURI_TARGET}"
cp "$EXE_PATH" "$OUTPUT_DIR/$TAURI_EXE_NAME"
echo "[OK] Copied to Tauri sidecar name: $TAURI_EXE_NAME"

echo ""
echo "=== Sidecar bundle complete ==="

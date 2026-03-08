#!/usr/bin/env bash
set -euo pipefail

echo "=== AI DJ Assist - Dev Environment Setup ==="

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "[OK] Node.js $NODE_VERSION"
else
    echo "[MISSING] Node.js >= 20 is required"
    exit 1
fi

# Check Rust
if command -v cargo &> /dev/null; then
    RUST_VERSION=$(rustc --version)
    echo "[OK] $RUST_VERSION"
else
    echo "[MISSING] Rust toolchain is required (install via rustup)"
    exit 1
fi

# Check Python
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version)
    echo "[OK] $PY_VERSION"
else
    echo "[MISSING] Python >= 3.11 is required"
    exit 1
fi

# Install Node dependencies
echo ""
echo "--- Installing Node.js dependencies ---"
npm install

# Set up Python virtual environment
echo ""
echo "--- Setting up Python environment ---"
cd packages/analysis
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -e ".[dev]"
deactivate
cd ../..

# Build Rust packages
echo ""
echo "--- Building Rust packages ---"
cargo build --workspace

# Initialize husky git hooks
echo ""
echo "--- Setting up git hooks (husky) ---"
npx husky init 2>/dev/null || true

echo ""
echo "=== Setup complete! ==="
echo "Run 'npm run dev' to start the desktop app."

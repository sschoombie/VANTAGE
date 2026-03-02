#!/usr/bin/env bash

# ============================
# Setup script for VANTAGE
# ============================

# ===== Configuration =====
VENV_NAME="venv_vantage"
REQUIRED_MAJOR="3"
REQUIRED_MINOR="12"

echo "Checking Python installation..."

# ===== Python candidates =====
PY_CANDIDATES=("python3.12" "python3" "python")
PY_FOUND=""

# ---- Find a Python 3.12.x installation ----
for p in "${PY_CANDIDATES[@]}"; do
    if command -v "$p" >/dev/null 2>&1; then
        VERSION=$("$p" --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo "$VERSION" | cut -d. -f1)
        MINOR=$(echo "$VERSION" | cut -d. -f2)
        echo "Checking Python at $p: version $VERSION"

        if [ "$MAJOR" = "$REQUIRED_MAJOR" ] && [ "$MINOR" = "$REQUIRED_MINOR" ]; then
            PY_FOUND="$p"
            break
        fi
    fi
done

if [ -z "$PY_FOUND" ]; then
    echo ""
    echo "No Python 3.12 installation found in PATH."
    echo "Please install Python 3.12.x from:"
    echo "https://www.python.org/downloads/release/python-3122/"
    echo ""
    echo "On macOS, you can also use Homebrew:"
    echo "brew install python@3.12"
    exit 1
fi

echo "Using Python: $PY_FOUND"

# ===== Check FFmpeg =====
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo ""
    echo "FFmpeg is not installed or not in PATH."
    echo "Install via your package manager, e.g.:"
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  Fedora:        sudo dnf install ffmpeg"
    echo "  Arch:          sudo pacman -S ffmpeg"
    echo "  macOS (brew):  brew install ffmpeg"
    exit 1
fi

echo "FFmpeg is installed."

# ===== Create virtual environment =====
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment..."
    "$PY_FOUND" -m venv "$VENV_NAME"
else
    echo "Virtual environment already exists."
fi

# ===== Activate venv and install dependencies =====
source "$VENV_NAME/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found."
    deactivate
    exit 1
fi

# ===== Finish =====
deactivate

echo ""
echo "Setup complete!"
echo "You can now run your script using run.sh"
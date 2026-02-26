#!/bin/bash

# ===== Required versions =====
REQUIRED_PYTHON="3.12.2"
VENV_NAME="venv_vantage"

echo "Checking Python installation..."

PY_FOUND=""

# ---- Find python executables in PATH ----
for p in $(which -a python3 2>/dev/null); do
    VERSION=$("$p" --version 2>&1 | awk '{print $2}')
    echo "Checking Python at $p: version $VERSION"
    if [ "$VERSION" == "$REQUIRED_PYTHON" ]; then
        PY_FOUND="$p"
        break
    fi
done

if [ -z "$PY_FOUND" ]; then
    echo ""
    echo "No Python $REQUIRED_PYTHON installation found in PATH."
    echo "Please install Python $REQUIRED_PYTHON from:"
    echo "https://www.python.org/downloads/release/python-3122/"
    echo ""
    echo "If using Homebrew:"
    echo "brew install python@3.12"
    exit 1
fi

echo "Using Python: $PY_FOUND"

# ===== Check ffmpeg =====
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo ""
    echo "FFmpeg is not installed or not in PATH."
    echo "Install it from:"
    echo "https://ffmpeg.org/download.html"
    echo ""
    echo "Or via Homebrew:"
    echo "brew install ffmpeg"
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

deactivate

echo ""
echo "Setup complete! You can now run your script."
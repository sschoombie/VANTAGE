#!/bin/bash

VENV_NAME="venv_vantage"
SCRIPT_NAME="VANTAGE_MAIN.py"

if [ ! -d "$VENV_NAME" ]; then
    echo "Virtual environment not found. Run setup.sh first."
    exit 1
fi

# ===== Activate venv =====
source "$VENV_NAME/bin/activate"

# ===== Run the Python script =====
echo "Running $SCRIPT_NAME..."
python "$SCRIPT_NAME"

# ===== Deactivate =====
deactivate
#!/usr/bin/env bash

# ============================
# Run script for VANTAGE
# ============================

VENV_NAME="venv_vantage"
SCRIPT_NAME="VANTAGE_MAIN.py"

# ===== Check that venv exists =====
if [ ! -d "$VENV_NAME" ]; then
    echo "Virtual environment '$VENV_NAME' not found."
    echo "Please run setup.sh first to create it."
    exit 1
fi

# ===== Activate virtual environment =====
source "$VENV_NAME/bin/activate"

# ===== Run the Python script =====
echo "Running $SCRIPT_NAME..."
python "$SCRIPT_NAME"

# ===== Deactivate virtual environment =====
deactivate

echo ""
echo "Script execution finished."
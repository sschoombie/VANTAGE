@echo off
setlocal

set VENV_NAME=venv_vantage
set SCRIPT_NAME=VANTAGE_MAIN.py

IF NOT EXIST %VENV_NAME% (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b
)

REM ===== Activate venv =====
call %VENV_NAME%\Scripts\activate

REM ===== Run the Python script =====
echo Running %SCRIPT_NAME%...
python %SCRIPT_NAME%

REM ===== Deactivate =====
deactivate

pause